//! The orbit camera, and how it gets from the pointer to the picture.
//!
//! The **main thread owns the camera**: pointer drags and the wheel change its
//! yaw / pitch / radius there, with the clamps below applied there. Once per
//! main-thread animation frame, if it changed, main posts the whole state
//! stamped with that frame's absolute time ([`CameraMsg::Pose`]).
//!
//! The render worker keeps the last few of those and, each frame, uses the
//! newest one stamped at least half a refresh period before its own frame
//! time ([`CameraFeed::pick`]). Main's frame N and the render worker's frame N
//! share the same frame time, so that rule always picks main's frame N−1 — a
//! state that was posted a whole frame earlier and has certainly arrived. Each
//! drawn frame therefore shows exactly one frame's worth of camera movement.
//! Applying each pointer event as it arrived instead raced the two threads'
//! frames: some frames got no movement and the next got two, a visible judder
//! while orbiting. The cost is one frame of camera latency.

use std::collections::VecDeque;

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Main → render worker: the camera state as of one main-thread frame.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum CameraMsg {
    Pose {
        /// Absolute frame time (`performance.timeOrigin + rAF timestamp`), ms.
        t_ms: f64,
        yaw: f32,
        pitch: f32,
        radius: f32,
    },
}

/// A minimal orbit camera: drag to orbit, wheel to dolly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub look_at: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            look_at: Vec3::new(-0.1, 0.45, 0.0),
            radius: 2.7,
            yaw: 0.45,
            pitch: 0.22,
        }
    }
}

impl OrbitCamera {
    const SENSITIVITY: f32 = 0.008;
    const PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const PITCH_MIN: f32 = 0.02;
    const MIN_RADIUS: f32 = 0.8;
    const MAX_RADIUS: f32 = 20.0;

    /// A drag of `dx`, `dy` CSS pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * Self::SENSITIVITY;
        self.pitch = (self.pitch - dy * Self::SENSITIVITY).clamp(Self::PITCH_MIN, Self::PITCH_MAX);
    }

    /// A wheel movement of `dy` (positive = away).
    pub fn zoom(&mut self, dy: f32) {
        self.radius = (self.radius * (1.0 + dy * 0.001)).clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.look_at
            + Vec3::new(
                self.radius * cp * sy,
                self.radius * sp,
                self.radius * cp * cy,
            )
    }
}

/// States kept on the render worker; a few frames' worth is plenty.
const FEED_LEN: usize = 4;

/// The render worker's side: the last few posted states.
#[derive(Default)]
pub struct CameraFeed {
    poses: VecDeque<CameraMsg>,
}

impl CameraFeed {
    pub fn push(&mut self, msg: CameraMsg) {
        if self.poses.len() == FEED_LEN {
            self.poses.pop_front();
        }
        self.poses.push_back(msg);
    }

    /// Applies to `camera` the newest state stamped no later than
    /// `frame_t_ms - period_ms / 2`, if there is one (else the camera keeps
    /// what it has).
    pub fn pick(&self, camera: &mut OrbitCamera, frame_t_ms: f64, period_ms: f64) {
        let cutoff = frame_t_ms - period_ms * 0.5;
        let newest = self.poses.iter().rev().find(|m| match m {
            CameraMsg::Pose { t_ms, .. } => *t_ms <= cutoff,
        });
        if let Some(CameraMsg::Pose {
            yaw, pitch, radius, ..
        }) = newest
        {
            camera.yaw = *yaw;
            camera.pitch = *pitch;
            camera.radius = *radius;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(t_ms: f64, yaw: f32) -> CameraMsg {
        CameraMsg::Pose {
            t_ms,
            yaw,
            pitch: 0.3,
            radius: 2.0,
        }
    }

    #[test]
    fn pick_uses_the_previous_main_frame() {
        let mut feed = CameraFeed::default();
        let mut cam = OrbitCamera::default();
        // Main frames at 0, 16.7, 33.3; the render frame at 33.3 must use the
        // state from 16.7 even though 33.3's has (just) arrived.
        feed.push(pose(0.0, 1.0));
        feed.push(pose(16.7, 2.0));
        feed.push(pose(33.3, 3.0));
        feed.pick(&mut cam, 33.3, 16.7);
        assert_eq!(cam.yaw, 2.0);
        // Nothing old enough: the camera keeps its state.
        let mut cam2 = OrbitCamera::default();
        let before = cam2.yaw;
        let mut late = CameraFeed::default();
        late.push(pose(33.3, 9.0));
        late.pick(&mut cam2, 33.3, 16.7);
        assert_eq!(cam2.yaw, before);
    }

    #[test]
    fn clamps_live_with_the_owner() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, -100_000.0);
        assert!(cam.pitch <= OrbitCamera::PITCH_MAX);
        // Reversing the drag moves at once: no dead zone past the clamp.
        let at_max = cam.pitch;
        cam.orbit(0.0, 10.0);
        assert!(cam.pitch < at_max);
        cam.zoom(1e9);
        assert_eq!(cam.radius, OrbitCamera::MAX_RADIUS);
    }
}
