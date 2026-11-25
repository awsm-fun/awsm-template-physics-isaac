//! What this app adds to the Stats panel (top right; the Stats chip, bottom
//! left, shows and hides it).
//!
//! awsm-renderer-stats does the measuring and the formatting: the frame
//! windows, the warm-up, the smoothing, the frame budget, the display's
//! refresh rate and missed frames, the renderer's CPU scopes and GPU passes
//! (its `StatsCollector`, in the render worker), and the panel that shows
//! the report (its `StatsPanel`, on the main thread). The panel tells the
//! render worker which groups are open (`stats-view`) and what only the page
//! can measure (`stats-page`: its own frame gaps and work); the render worker
//! posts a report about once a second.
//!
//! One group per thread, so that every number sits under the thread that
//! spent it, and every number says its unit and what it is per:
//!
//! ```text
//! frame budget               the display's refresh period, and what each lane
//!                            (the GPU, each measured thread) uses of it per
//!                            frame.
//! display                    the display's refresh rate (measured on the main
//!                            thread, from its own animation frames), the
//!                            frames the render worker presented per second,
//!                            and the refreshes that went by without a new
//!                            frame.
//! GPU                        the GPU's time per frame, from the renderer's
//!                            timestamp queries (`n/a` without them). Its rows
//!                            are the passes.
//! renderer (render worker)   the render worker's CPU time per frame: its five
//!                            phases ([`Phase`]), and under `render (encode +
//!                            submit)` awsm-renderer's own CPU scopes.
//! poses (…, pose worker)     this file: the pose source as the page sees it:
//!                            the samples per second each robot gets, the
//!                            frames drawn past a robot's newest sample
//!                            (late), and, when streaming, the frames dropped.
//!                            The sim itself (Isaac, on another machine) is not
//!                            measurable from here, so it has no step time and
//!                            no lane of the frame budget.
//! page (main thread)         the main thread's own work per frame (`camera
//!                            frame loop`).
//! ```
//!
//! **Late frames.** Each frame shows every robot at the frame's own time,
//! between the two samples around it (see `protocol`, "Timing"). A frame whose
//! time is past a robot's newest sample holds that pose instead: the robot
//! pauses. Replay publishes ahead and is never late; a stream is late when it
//! stalls or its jitter outruns the playout delay. Late frames in
//! `LATE_WINDOWS` windows in a row turn the row and the pose header red.

use awsm_renderer_stats::{AppSection, SectionSpec, StatRow, StatsCollector, StatsConfig, Window};
use wasm_bindgen::JsCast;
use web_sys::js_sys;

/// The frame function's phases, in the order they run.
#[derive(Clone, Copy)]
pub enum Phase {
    /// The scene's authored clips, texture UV flows and particles
    /// (`update_animations`, `ParticleDriver::tick`).
    Animate,
    /// Reading and blending every robot's two samples and writing its geom
    /// transforms: after `Animate`, so the poses win on their nodes.
    Read,
    Camera,
    Transforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
}

const PHASE_NAMES: [&str; 5] = [
    "animations + particles",
    "read app state + interpolate",
    "set camera",
    "update transforms",
    "render (encode + submit)",
];

/// The pose group's id: "sim", the id every template gives its simulation
/// (the page stores whether it is open under it).
pub const POSES: &str = "sim";

fn poses_label(stream: bool) -> &'static str {
    if stream {
        "poses (live stream, pose worker)"
    } else {
        "poses (replay, pose worker)"
    }
}

/// The page's work, as the page group names it.
pub const PAGE_PART: &str = "camera frame loop";

/// The panel's configuration, the same on both threads (the main thread
/// shows the "starting…" report from it before the render worker is up).
/// `stream`: the poses come from a live stream rather than the replay.
pub fn config(stream: bool) -> StatsConfig {
    StatsConfig {
        phases: PHASE_NAMES.map(String::from).to_vec(),
        render_phase: Some(Phase::Render as usize),
        // Coloured as a simulation; it has no lane of the frame budget.
        sections: vec![SectionSpec::new(POSES, poses_label(stream), Some("sim"))],
        // The renderer draws on every refresh: no pacing to report.
        pacing_row: false,
        ..StatsConfig::default()
    }
}

/// The pose group, from the pose block's running counters.
pub struct PoseStats {
    stream: bool,
    /// The counters (samples, dropped) at the last report, and when (the
    /// worker's clock): they run in real time, pauses included, so their
    /// rates are per real second.
    base: Option<(f64, u32, u32)>,
    dropped_rate: f64,
    dropped_total: u64,
}

impl PoseStats {
    pub fn new(stream: bool) -> Self {
        Self {
            stream,
            base: None,
            dropped_rate: 0.0,
            dropped_total: 0,
        }
    }

    /// Counting starts over (the scene is on screen; the collector's own
    /// figures restart too).
    pub fn restart(&mut self) {
        *self = Self::new(self.stream);
    }

    /// The group for one window. `counters`: the robots the block feeds, and
    /// its samples and dropped counters; `None` until the block is linked.
    pub fn section(
        &mut self,
        stats: &mut StatsCollector,
        window: &Window,
        now_ms: f64,
        counters: Option<(u32, u32, u32)>,
    ) -> AppSection {
        let robots = counters.map_or(0, |c| c.0);
        if let Some((_, samples, dropped)) = counters {
            if let Some((t0, samples0, dropped0)) = self.base {
                let seconds = (now_ms - t0) / 1000.0;
                if seconds > 0.0 {
                    let samples = samples.wrapping_sub(samples0);
                    let dropped = dropped.wrapping_sub(dropped0);
                    stats.smooth("samples", f64::from(samples) / seconds);
                    self.dropped_total += u64::from(dropped);
                    self.dropped_rate = f64::from(dropped) / seconds;
                }
            }
            self.base = (robots > 0).then_some((now_ms, samples, dropped));
        }
        let samples = stats.smoothed("samples");

        let mut rows = vec![
            StatRow::new(
                "samples published",
                samples.map_or("—".to_string(), |r| {
                    format!("{r:.0} / s · {robots} robots")
                }),
            ),
            window.late.row("late frames"),
        ];
        if self.stream {
            rows.push(StatRow::new(
                "dropped stream frames",
                format!(
                    "{:.0} / s · {} total",
                    self.dropped_rate, self.dropped_total
                ),
            ));
        }
        let per_robot = samples
            .filter(|_| robots > 0)
            .map(|r| r / f64::from(robots));
        SectionSpec::new(POSES, poses_label(self.stream), Some("sim"))
            .group()
            .headline(per_robot.map_or("—".to_string(), |r| format!("{r:.0} samples/s per robot")))
            .bad(window.late.bad)
            .rows(rows)
    }
}

/// `performance.timeOrigin` in the render worker: added to a frame timestamp,
/// it gives the absolute time every thread of the page agrees on.
pub fn worker_time_origin() -> f64 {
    js_sys::global()
        .dyn_into::<web_sys::WorkerGlobalScope>()
        .ok()
        .and_then(|g| g.performance())
        .map(|p| p.time_origin())
        .unwrap_or(0.0)
}

/// `performance.now()` in the render worker.
pub fn worker_now() -> f64 {
    js_sys::global()
        .dyn_into::<web_sys::WorkerGlobalScope>()
        .ok()
        .and_then(|g| g.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use awsm_renderer_stats::LateStats;

    fn window() -> Window {
        Window {
            seconds: 1.0,
            frames: 60,
            span_ms: 1000.0,
            late: LateStats::default(),
        }
    }

    #[test]
    fn pose_rates_are_per_real_second_and_per_robot() {
        let mut collector = StatsCollector::new(config(true));
        let mut poses = PoseStats::new(true);
        let first = poses.section(&mut collector, &window(), 0.0, Some((2, 0, 0)));
        assert_eq!(first.headline, "—");
        let next = poses.section(&mut collector, &window(), 1000.0, Some((2, 60, 3)));
        assert_eq!(next.headline, "30 samples/s per robot");
        assert_eq!(next.rows[0].value, "60 / s · 2 robots");
        assert_eq!(next.rows[2].value, "3 / s · 3 total");
        assert_eq!(next.label, "poses (live stream, pose worker)");
        assert!(next.budget_ms.is_none());
    }
}
