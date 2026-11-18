//! The wire between the players:
//!
//! - **pose worker → render worker**: a plain `SharedArrayBuffer` pose block
//!   (layout below). The pose worker (`web/workers/pose-worker.js`) publishes
//!   every robot's geom world poses as time-stamped samples; each frame the
//!   render thread interpolates every robot at its frame time and hands the
//!   result to the renderer's pose sink. The renderer knows nothing about Isaac, captures or sockets.
//! - **main ↔ render worker**: small serde messages (progress, the scene's sim
//!   instances, resize, the Stats report and which Stats groups are open)
//!   and, once per main-thread frame while it moves, the camera state (see
//!   `crate::camera`).
//! - **main ↔ pose worker**: raw JS objects (`{kind: ...}`) — see the worker.
//!
//! ## Where the poses come from
//!
//! Isaac Sim cannot run in a browser, so the pose worker has two sources and
//! the rest of the app cannot tell them apart:
//!
//! - **replay** (default): loops each robot's `media/captures/<model>.capture.json`
//!   — a scripted joint sweep recorded offline by forward kinematics over the
//!   USD joints (`awsm-renderer-isaac-record`). Joint animation, not physics.
//! - **stream** (`?stream=ws://host:port`): frames from a live producer — an
//!   Isaac Lab harness (`harness/isaac_lab_stream.py`) or the mock that
//!   replays captures over a socket (`harness/replay_stream.py`).
//!
//! ## Stream frames (the WebSocket wire)
//!
//! One binary message per robot per step, in either of two forms:
//!
//! ```text
//! bytes 0..32   SHA-256 of the robot's root USD file — the scene instance's
//!               fingerprint (`source.sha256`), raw bytes
//! stepped form (recommended):
//!   bytes 32..36  u32 LE  the producer's step number (monotonic)
//!   bytes 36..40  f32 LE  seconds per step (dt)
//!   bytes 40..    7 × geom_count little-endian f32 poses
//! plain form (older producers):
//!   bytes 32..    7 × geom_count little-endian f32 poses
//! ```
//!
//! poses are `[px,py,pz,qw,qx,qy,qz]` per geom, in the sidecar's geom order
//! (Z-up, metres, env-local). The two forms differ in length by 8 bytes, which
//! is never a multiple of a geom's 28, so the length says which one arrived.
//!
//! The fingerprint prefix IS the binding: a frame for a model the scene does
//! not contain, or of the wrong length, is dropped and counted, never applied
//! to whichever robot happens to be first. Several robots share one socket.
//!
//! ## Timing: every sample carries the time it is shown at
//!
//! The renderer never shows "whatever arrived last". Every pose sample in the
//! block carries an absolute time (`performance.timeOrigin + now`, ms — the
//! same clock in every thread of the page), and each frame the render worker
//! interpolates every robot between the two samples around its own frame
//! time: positions lerp, rotations nlerp. So motion is paced by the display's
//! clock, whatever the timers and the network did.
//!
//! - **replay**: the pose worker publishes each capture frame stamped with
//!   when it is due on the loop's timeline, a little ahead of time, so the
//!   render worker always has the pair around "now". The recorded motion is
//!   sampled exactly at each display frame, with no added latency.
//! - **stream**: a frame's time is `step · dt` mapped onto the page clock by
//!   the smallest recent (arrival − step · dt) — the least-delayed arrivals
//!   define the mapping — plus a playout delay of 2.5 steps, so ordinary
//!   network jitter never leaves the renderer without the next sample. A
//!   plain-form frame counts as one step, with dt estimated from arrivals.
//!
//! If the renderer's frame time is past the newest sample (the source stalled
//! or the network spiked beyond the playout delay), the robot holds the newest
//! pose; it never extrapolates.
//!
//! ## Pose block layout (the SharedArrayBuffer)
//!
//! ```text
//! header (16 bytes)
//!   i32[0]  layout   BLOCK_LAYOUT (2)
//!   i32[1]  frames   samples published, all robots (monotonic)
//!   i32[2]  source   0 = waiting, 1 = replay, 2 = live stream
//!   i32[3]  dropped  stream frames rejected (unknown fingerprint / bad length)
//! then, per robot in the scene's instance order, a ring of RING_SLOTS slots:
//!   slot (slot_bytes(geoms) bytes, 8-byte aligned)
//!     i32  seq    the slot's seqlock: ODD while being written, EVEN when stable
//!     i32  —
//!     f64  t_ms   absolute time this sample is shown at
//!     f32  poses  7 × geoms: [px,py,pz,qw,qx,qy,qz] per geom
//! ```
//!
//! One writer (the pose worker) fills each robot's slots in turn; one reader
//! (the render worker) reads seq, then the slot, then seq again, and ignores a
//! slot that was being written. It never waits: the writer is always several
//! slots away from the pair the reader needs.
//!
//! Poses are raw sim world poses in the documented pose-sink convention —
//! nothing is reshaped on either side of the block.

use serde::{Deserialize, Serialize};

/// f32 values per geom: pos xyz + quat wxyz.
pub const POSE_STRIDE: usize = 7;
/// Byte size of the block header.
pub const HEADER_BYTES: usize = 16;
/// `i32[0]` of the header: the layout this code reads and writes.
pub const BLOCK_LAYOUT: i32 = 2;
/// Header slot: samples published, all robots.
pub const HEADER_FRAMES: u32 = 1;
/// Header slot: what is feeding the block.
pub const HEADER_SOURCE: u32 = 2;
/// Header slot: rejected stream frames.
pub const HEADER_DROPPED: u32 = 3;
/// Samples kept per robot. At 30 Hz that is 266 ms of motion: the pair the
/// renderer needs, the replay's lead and the stream's playout delay fit with
/// room to spare, so the writer never reaches the slots being read.
pub const RING_SLOTS: usize = 8;
/// Byte offset of the poses inside a slot (after seq, padding and t_ms).
pub const SLOT_POSES: usize = 16;

/// Byte size of one ring slot for a robot with `geoms` geoms (8-aligned, so
/// every slot's `t_ms` is an aligned f64).
pub fn slot_bytes(geoms: usize) -> usize {
    (SLOT_POSES + geoms * POSE_STRIDE * 4).div_ceil(8) * 8
}

/// Byte offset of each robot's ring, and the total size of the block, for
/// robots with `geoms` geoms each.
pub fn ring_offsets(geoms: &[usize]) -> (Vec<usize>, usize) {
    let mut at = HEADER_BYTES;
    let offsets = geoms
        .iter()
        .map(|g| {
            let here = at;
            at += RING_SLOTS * slot_bytes(*g);
            here
        })
        .collect();
    (offsets, at)
}

/// Byte size of the pose block for robots with `geoms` geoms each.
pub fn pose_block_bytes(geoms: &[usize]) -> usize {
    ring_offsets(geoms).1
}

/// One sim instance in the loaded scene, as the pose worker needs to know it.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InstanceInfo {
    /// The compiled model's name (`panda`, `anymal`) — the replay capture is
    /// `media/captures/<model>.capture.json`.
    pub model: String,
    /// Lowercase hex SHA-256 of the root USD — the fingerprint every capture
    /// and stream frame must carry.
    pub sha256: String,
    pub geoms: usize,
}

/// Render-worker → main messages.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum RenderMsg {
    /// Human-readable loading-screen line.
    Progress {
        message: String,
    },
    /// The scene bundle is loaded and committed. `instances` are its sim
    /// instances in order — the layout of the pose block.
    SceneReady {
        instances: Vec<InstanceInfo>,
    },
    /// First real frames have been presented.
    Ready,
    /// About once a second, for the Stats panel (see `crate::stats`).
    Stats(FrameReport),
    Error {
        message: String,
    },
}

/// The render worker's Stats report: one window of about a second.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct FrameReport {
    /// Frames presented, and the time they covered (rAF timestamp to rAF
    /// timestamp, pauses left out).
    pub frames: u32,
    pub span_ms: f64,
    /// The render worker's CPU time for those frames (the whole frame function).
    pub cpu_ms: f64,
    /// Frames whose time was past a robot's newest sample, which held its
    /// newest pose instead (the source stalled or fell behind).
    pub late: u32,
    /// Whether the device has timestamp queries (else no GPU numbers, ever).
    pub gpu_available: bool,
    /// Mean GPU time per frame over the window, when there was a sample.
    pub gpu_ms: Option<f64>,
    /// Only while the renderer group is open: the mean per frame of each of
    /// the frame function's phases (`crate::stats::Phase` order), and of each
    /// of awsm-renderer's CPU scopes inside `render()` (named as the renderer
    /// names them; the panel groups them with `cpu_scope_kind`).
    pub phases_ms: Option<[f64; crate::stats::PHASES]>,
    pub cpu_scopes: Vec<(String, f64)>,
    /// Only while the GPU group is open: the mean per frame of each GPU pass.
    pub gpu_passes: Vec<(String, f64)>,
    /// The pose block's running counters (header `frames` and `dropped`), and
    /// the robots it feeds; zero until the block is linked.
    pub robots: u32,
    pub samples: u32,
    pub dropped: u32,
}

/// Which Stats groups are open (and the panel shown): the render worker runs
/// the extra timing for a group only then.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Profile {
    pub renderer: bool,
    pub gpu: bool,
}

/// Main → render worker: see [`Profile`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum ProfileMsg {
    Set(Profile),
}

/// Main → render worker: canvas backing-store resize (device pixels).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResizeMsg {
    Canvas { width: u32, height: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_holds_a_ring_per_robot_after_the_header() {
        assert_eq!(pose_block_bytes(&[]), HEADER_BYTES);
        // 16 + 44·28 = 1248 (already 8-aligned); 16 + 63·28 = 1780 → 1784.
        assert_eq!(slot_bytes(44), 1248);
        assert_eq!(slot_bytes(63), 1784);
        let (offsets, total) = ring_offsets(&[44, 63]);
        assert_eq!(offsets, vec![16, 16 + 8 * 1248]);
        assert_eq!(total, 16 + 8 * 1248 + 8 * 1784);
        assert!(offsets.iter().all(|o| o % 8 == 0));
    }
}
