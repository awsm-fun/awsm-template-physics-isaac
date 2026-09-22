//! The wire between the players:
//!
//! - **pose worker → render worker**: a plain `SharedArrayBuffer` pose block
//!   (layout below). The pose worker (`web/workers/pose-worker.js`) publishes
//!   every robot's geom world poses; the render thread reads the latest stable
//!   snapshot per frame and hands each robot's span to the renderer's pose
//!   sink. The renderer knows nothing about Isaac, captures or sockets.
//! - **main ↔ render worker**: small serde messages (progress, the scene's sim
//!   instances, resize, camera gestures) — none of them per-frame.
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
//! One binary message per robot per step:
//!
//! ```text
//! bytes 0..32   SHA-256 of the robot's root USD file — the scene instance's
//!               fingerprint (`source.sha256`), raw bytes
//! bytes 32..    7 × geom_count little-endian f32: [px,py,pz,qw,qx,qy,qz] per
//!               geom, in the sidecar's geom order (Z-up, metres, env-local)
//! ```
//!
//! The fingerprint prefix IS the binding: a frame for a model the scene does
//! not contain, or of the wrong length, is dropped and counted, never applied
//! to whichever robot happens to be first. Several robots share one socket.
//!
//! ## Pose block layout (the SharedArrayBuffer)
//!
//! ```text
//! i32[0]  seq      seqlock: writer makes it ODD, writes poses, makes it EVEN
//! i32[1]  frames   frames published (monotonic)
//! i32[2]  source   0 = waiting, 1 = replay, 2 = live stream
//! i32[3]  dropped  stream frames rejected (unknown fingerprint / bad length)
//! f32[4 + (o_i + g)*7 + 0..3]   robot i, geom g: world position
//! f32[4 + (o_i + g)*7 + 3..7]   robot i, geom g: rotation [w, x, y, z]
//!     where o_i = geoms of robots 0..i, in the scene's instance order
//! ```
//!
//! Poses are raw sim world poses in the documented pose-sink convention — the
//! block is a stream frame, so nothing is reshaped on either side of it.

use serde::{Deserialize, Serialize};

/// f32 slots per geom in the pose block: pos xyz + quat wxyz.
pub const POSE_STRIDE: usize = 7;
/// i32/f32 header slots before the pose span.
pub const POSE_HEADER: usize = 4;

/// Header slot: what is feeding the block.
pub const HEADER_SOURCE: u32 = 2;
/// Header slot: rejected stream frames.
pub const HEADER_DROPPED: u32 = 3;

/// Byte size of the pose block for robots with `geoms` geoms each.
pub fn pose_block_bytes(geoms: &[usize]) -> usize {
    (POSE_HEADER + geoms.iter().sum::<usize>() * POSE_STRIDE) * 4
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
    Error {
        message: String,
    },
}

/// Main → render worker: canvas backing-store resize (device pixels).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResizeMsg {
    Canvas { width: u32, height: u32 },
}

/// Main → render worker: orbit-camera gestures.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CameraMsg {
    Orbit { dx: f32, dy: f32 },
    Zoom { dy: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_block_holds_every_robot_after_the_header() {
        assert_eq!(pose_block_bytes(&[]), 16);
        assert_eq!(pose_block_bytes(&[44, 64]), (4 + 108 * 7) * 4);
    }
}
