//! Render thread — hosts `awsm-renderer` on an `OffscreenCanvas`, loads the
//! editor-exported `scene.toml` through the **player** path, and drives every
//! robot in it from the pose worker's shared pose block.
//!
//! The robots are *authored content*: exported from their USD assets by
//! `awsm-renderer-isaac-export`, imported in the editor, and shipped in the
//! bundle like any other geometry — so this worker builds nothing. It:
//!
//! 1. reports the scene's sim instances (model name, fingerprint, geom count)
//!    to main, which sizes the pose block for exactly those robots;
//! 2. each frame, interpolates every robot at the frame's own time between the
//!    two time-stamped samples around it in the robot's ring (see
//!    [`crate::protocol`]), and hands the result to
//!    `scene_loader::mujoco::apply_geom_poses`. Rendering is paced by the
//!    display's clock alone: how evenly the pose worker's timers or the network
//!    delivered the samples never shows.
//!
//! The pose worker already refused any capture or stream frame whose
//! fingerprint does not match its robot, so what arrives here is bound.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use glam::{Mat4, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use awsm_renderer::AwsmRenderer;
use awsm_renderer_settings::{
    apply_anti_aliasing, post_to_page, scene_anti_aliasing, AaMode, SettingsToPage,
    SettingsToWorker,
};
use awsm_renderer_stats::{
    post_report, stats_logging, FrameInput, RendererTimings, StatsCollector, StatsToWorker,
};

use crate::camera::{CameraFeed, CameraMsg, OrbitCamera};
use crate::protocol::{
    ring_offsets, slot_bytes, InstanceInfo, RenderMsg, HEADER_DROPPED, HEADER_FRAMES, POSE_STRIDE,
    RING_SLOTS, SLOT_POSES,
};
use crate::stats::{self, worker_now, Phase, PoseStats};

/// The boxed RAF callback, self-referenced so the render loop can reschedule
/// itself each frame (and stays alive for the worker's lifetime).
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Lens + clip planes for the one camera this app has.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.05;
const CAMERA_FAR: f32 = 400.0;

/// The awsm-renderer release the vendored Basis transcoder files came with.
/// It is appended to their URLs as `?v=…`: those files keep their names from
/// one release to the next, so after a deployed upgrade a browser could
/// otherwise pair a cached old transcoder with the new worker script. Bump it
/// together with the awsm-renderer version in Cargo.toml.
const BASIS_VERSION: &str = "0.46.0";

/// The renderer, shared by the frame loop and the message handler. `None`
/// while an anti-aliasing change owns it across its awaits.
type RendererCell = Rc<RefCell<Option<AwsmRenderer>>>;

/// The live link between the pose block and the scene's robots.
struct PoseLink {
    robots: Vec<RobotRing>,
    /// i32 view over the whole block: the slots' seq words.
    words: js_sys::Int32Array,
    /// f64 view over the whole block: the slots' times.
    times: js_sys::Float64Array,
}

/// One robot's ring of time-stamped samples in the block.
struct RobotRing {
    /// The loader-resolved instance the poses drive.
    instance: awsm_renderer_scene_loader::mujoco::MujocoInstance,
    /// Byte offset of slot 0, and the byte size of a slot.
    base: usize,
    slot_bytes: usize,
    /// f32 view of each slot's poses (built once, so a frame makes no views).
    poses: Vec<js_sys::Float32Array>,
    /// Copy targets for the two samples, and the blend of them.
    a: Vec<f32>,
    b: Vec<f32>,
    out: Vec<f32>,
    /// What was applied last frame (slot seqs + blend factor): unchanged means
    /// nothing to do.
    last: Option<(i32, i32, u32)>,
}

/// A refresh-period estimate from the render loop's own frame intervals (the
/// median of the recent ones), for the camera's selection rule.
struct FramePeriod {
    prev: Option<f64>,
    gaps: std::collections::VecDeque<f64>,
}

impl FramePeriod {
    fn new() -> Self {
        Self {
            prev: None,
            gaps: std::collections::VecDeque::new(),
        }
    }

    /// Records a frame at `t` and returns the period (ms).
    fn frame(&mut self, t: f64) -> f64 {
        if let Some(prev) = self.prev.replace(t) {
            let gap = t - prev;
            if gap > 0.0 && gap < 100.0 {
                if self.gaps.len() == 16 {
                    self.gaps.pop_front();
                }
                self.gaps.push_back(gap);
            }
        }
        let mut sorted: Vec<f64> = self.gaps.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        sorted
            .get(sorted.len() / 2)
            .copied()
            .unwrap_or(1000.0 / 60.0)
    }
}

/// Worker entry: unpack the transferred `OffscreenCanvas`, the page origin and
/// the page's settings (`aa`, the anti-aliasing mode id; `stream`, whether the
/// poses come from a live stream), build the WebGPU device, and kick off the
/// async load + render.
pub fn start(payload: JsValue) -> Result<(), JsValue> {
    use awsm_renderer::core::renderer::{AwsmRendererWebGpuBuilder, DeviceRequestLimits};
    use awsm_renderer::web_global::navigator_gpu;
    use awsm_renderer_scene_loader::basis::{configure, BasisWorkerConfig};

    let canvas: web_sys::OffscreenCanvas =
        js_sys::Reflect::get(&payload, &JsValue::from_str("canvas"))?.unchecked_into();
    // The worker has a `blob:` base URL, so relative fetches can't resolve —
    // main passes the media origin so we can build the absolute scene.toml URL.
    let origin = js_sys::Reflect::get(&payload, &JsValue::from_str("origin"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    // The mode the page's Settings chose: the renderer is built with it, so
    // start-up compiles exactly the pipelines this session uses.
    let aa = js_sys::Reflect::get(&payload, &JsValue::from_str("aa"))
        .ok()
        .and_then(|v| v.as_string())
        .and_then(|id| AaMode::from_id(&id))
        .unwrap_or(AaMode::Msaa);
    let stream = js_sys::Reflect::get(&payload, &JsValue::from_str("stream"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Provide the Basis (KTX2/BasisU) codec URLs — the crate hardcodes none.
    // Must be ABSOLUTE (our base is `blob:`); `app_base` keeps the deploy PATH
    // so a subpath deploy resolves.
    {
        let app_base = js_sys::Reflect::get(&payload, &JsValue::from_str("app_base"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let base = app_base.trim_end_matches('/');
        configure(BasisWorkerConfig::player(
            // `?v=` names the vendored build: these files keep their names
            // across releases, so a browser that cached an older transcoder
            // (wasm included: the worker carries the query over to it) would
            // otherwise pair it with the new JS. See `BASIS_VERSION`.
            format!("{base}/workers/basis-worker.js?v={BASIS_VERSION}"),
            format!("{base}/vendor/basis/basis_transcoder.js?v={BASIS_VERSION}"),
        ));
    }
    let canvas_handle = canvas.clone();
    post_progress("render worker: requesting WebGPU device…");
    let gpu =
        navigator_gpu().ok_or_else(|| JsValue::from_str("render worker: no navigator.gpu"))?;
    let gpu_builder = AwsmRendererWebGpuBuilder::new_with_offscreen_canvas(gpu, canvas)
        .with_device_request_limits(DeviceRequestLimits::max_all());

    wasm_bindgen_futures::spawn_local(async move {
        if let Err(err) = run(gpu_builder, canvas_handle, origin, aa, stream).await {
            tracing::error!("render thread: {err:?}");
            post_to_main(&RenderMsg::Error {
                message: format!("{err:?}"),
            });
        }
    });
    Ok(())
}

async fn run(
    gpu_builder: awsm_renderer::core::renderer::AwsmRendererWebGpuBuilder,
    canvas: web_sys::OffscreenCanvas,
    origin: String,
    aa: AaMode,
    stream: bool,
) -> Result<(), JsValue> {
    use awsm_renderer::camera::CameraParams;
    use awsm_renderer::AwsmRendererBuilder;

    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(aa.to_anti_aliasing())
        // GPU frame timing for the Stats panel's GPU group: timestamp queries
        // on every pass, folded into one whole-frame span (cheap; silently
        // off without `timestamp-query`). The CPU tier stays `Off` until the
        // renderer group is expanded.
        .with_logging(stats_logging())
        .with_phase_handler(|phase| {
            use awsm_renderer::RendererLoadingPhase as P;
            let message = match phase {
                P::Init => "renderer init: GPU device + core resources…",
                P::CompilingShaders | P::BuildingPipelines | P::Ready => return,
            };
            post_progress(message);
        })
        .build()
        .await
        .map_err(|e| JsValue::from_str(&format!("build renderer: {e}")))?;
    post_progress(&format!(
        "render worker: WebGPU device + renderer ready (anti-aliasing: {})",
        AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
    ));

    // ── Warm-up + scene fetch, CONCURRENTLY ─────────────────────────────────
    let bundle_base = format!("{}/bundle", origin.trim_end_matches('/'));
    let scene_url = format!("{bundle_base}/scene.toml");
    post_progress("compiling core render pipelines + fetching scene… (first visit can take a while — cached after)");
    let (compiled, scene) =
        futures::join!(renderer.ensure_config_pipelines(), fetch_scene(&scene_url));
    let compiled =
        compiled.map_err(|e| JsValue::from_str(&format!("ensure_config_pipelines: {e}")))?;
    let mut scene =
        scene.map_err(|e| JsValue::from_str(&format!("load scene {scene_url}: {e}")))?;
    // `load_scene_for_player` applies the scene's authored anti-aliasing
    // (`post_process.anti_aliasing`): make it the renderer's, the page's
    // choice, so the loader keeps it and nothing compiles twice.
    scene.post_process.anti_aliasing = scene_anti_aliasing(&renderer.anti_aliasing);
    post_progress(&format!(
        "core pipelines ready ({compiled}) · scene parsed ({} nodes)",
        scene.nodes.len()
    ));

    let assets = awsm_renderer_scene_loader::assets::HttpAssets::new(bundle_base.clone());
    let mut last_phase_line = String::new();
    let loaded = awsm_renderer_scene_loader::load_scene_for_player(
        &mut renderer,
        &scene,
        &assets,
        |phase| {
            let line = phase.label();
            if line != last_phase_line && !line.contains("0/0") {
                post_progress(&line);
                last_phase_line = line;
            }
        },
    )
    .await
    .map_err(|e| JsValue::from_str(&format!("load_scene_for_player: {e}")))?;

    // The loader builds the scene's particle emitters; the host ticks them
    // (as it does clips). Built before `loaded.mujoco` is moved out below.
    let mut particles = awsm_renderer_scene_loader::ParticleDriver::new(&loaded);

    // Every robot is authored content: the loader already resolved each sim
    // instance's geom_id → transform binding out of the bundle.
    let instances = loaded.mujoco;
    if instances.is_empty() {
        return Err(JsValue::from_str(
            "the bundle has no sim instances — export a scene with robots imported \
             via import_mujoco_from_url (see the README)",
        ));
    }
    let infos: Vec<InstanceInfo> = instances
        .iter()
        .map(|i| InstanceInfo {
            model: i.model_name.clone().unwrap_or_else(|| {
                i.source
                    .filename
                    .rsplit_once('.')
                    .map_or(i.source.filename.clone(), |(stem, _)| stem.to_string())
            }),
            sha256: i.source.sha256.clone(),
            geoms: i.geoms.len(),
        })
        .collect();
    for i in &infos {
        tracing::info!(
            "render thread: robot {} ({} geoms, {})",
            i.model,
            i.geoms,
            &i.sha256[..12.min(i.sha256.len())]
        );
    }

    let mut last_commit_line = String::new();
    renderer
        .commit_load(|stats| {
            let Some(line) = stats.phase_label() else {
                return;
            };
            if line != last_commit_line && !line.contains("0/0") {
                post_progress(&line);
                last_commit_line = line;
            }
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("commit_load: {e}")))?;
    renderer.update_transforms();
    post_progress("gpu commit complete — waiting for poses…");

    // ── Camera + shared state, then hand control to the event loop ──────────
    // The camera state comes from main (see `crate::camera`); until the first
    // one arrives, the same default main starts from.
    let camera = Rc::new(RefCell::new(OrbitCamera::default()));
    let feed = Rc::new(RefCell::new(CameraFeed::default()));
    // If the page asked for TAA and this device cannot run it, the renderer
    // fell back to MSAA 4× while loading: tell the page (it stops offering
    // TAA and shows what runs).
    if aa.is_taa() {
        if let Some(reason) = renderer.taa_fallback_reason() {
            tracing::warn!("render thread: TAA unavailable, running MSAA 4×: {reason}");
            post_to_page(&SettingsToPage::AaCompileDone {
                ok: true,
                error: None,
                mode: AaMode::from_anti_aliasing(&renderer.anti_aliasing),
                taa_fallback: Some(reason.to_string()),
            });
        }
    }
    #[allow(clippy::arc_with_non_send_sync)]
    let cell: RendererCell = Rc::new(RefCell::new(Some(renderer)));
    let link: Rc<RefCell<Option<PoseLink>>> = Rc::new(RefCell::new(None));
    let pending = Rc::new(RefCell::new(Some(instances)));
    // The Stats panel's numbers (awsm-renderer-stats): main tells it which
    // groups are open and what the page measured; it posts a report about
    // once a second.
    let collector = Rc::new(RefCell::new(StatsCollector::new(stats::config(stream))));
    // The latest anti-aliasing change the page asked for and that has not been
    // applied yet (a newer one replaces it), and whether one is being applied.
    let pending_aa: Rc<Cell<Option<AaMode>>> = Rc::new(Cell::new(None));
    let busy = Rc::new(Cell::new(false));

    install_onmessage(
        link.clone(),
        pending,
        feed.clone(),
        canvas.clone(),
        collector.clone(),
        pending_aa.clone(),
    )?;
    // Only now is the worker listening — main allocates the pose block from
    // this message.
    post_to_main(&RenderMsg::SceneReady { instances: infos });

    // ── Render loop ─────────────────────────────────────────────────────────
    #[allow(clippy::arc_with_non_send_sync)]
    let raf: RafCell = Rc::new(RefCell::new(None));
    let raf_init = raf.clone();
    let raf_run = raf.clone();
    let mut frame_count: u32 = 0;
    let mut poses = PoseStats::new(stream);
    let mut period = FramePeriod::new();
    // The page clock every thread shares (absolute ms): pose samples and
    // camera states are stamped on it.
    let time_origin = stats::worker_time_origin();
    let mut prev_vsync: Option<f64> = None;

    *raf_init.borrow_mut() = Some(Closure::new(move |vsync_ms: f64| {
        // An anti-aliasing change owns the renderer: skip the frame.
        if busy.get() {
            schedule(&raf_run);
            return;
        }
        // A pending Settings anti-aliasing change: apply it off the render
        // path. `apply_anti_aliasing` (set_anti_aliasing + commit_load) is
        // async and compiles the new mode's pipelines (the first switch in
        // each direction; later ones hit the renderer's cache). It owns the
        // renderer across its awaits, so no RefCell borrow is held over them;
        // `busy` keeps every frame off the renderer until it is back. The
        // page raises its "compiling pipelines" overlay from the events.
        if let Some(mode) = pending_aa.take() {
            if let Some(mut renderer) = cell.borrow_mut().take() {
                busy.set(true);
                let (cell, busy) = (cell.clone(), busy.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    let result = apply_anti_aliasing(&mut renderer, mode, |event| {
                        post_to_page(&SettingsToPage::from(event))
                    })
                    .await;
                    match result {
                        Ok(()) => tracing::info!(
                            "render thread: anti-aliasing {}",
                            AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
                        ),
                        Err(message) => {
                            tracing::error!("render thread: {message}");
                            post_to_main(&RenderMsg::Error { message });
                        }
                    }
                    *cell.borrow_mut() = Some(renderer);
                    busy.set(false);
                });
            }
            schedule(&raf_run);
            return;
        }

        let frame_start = worker_now();
        let mut guard = cell.borrow_mut();
        let Some(r) = guard.as_mut() else {
            schedule(&raf_run);
            return;
        };

        // The renderer group's breakdown (our phases and awsm-renderer's CPU
        // scopes) is timed on every frame while it is expanded, never else.
        let detail = {
            let mut c = collector.borrow_mut();
            c.begin_renderer_frame(r, frame_start);
            c.renderer_detail()
        };
        let lap = |phase: Phase| {
            if detail {
                collector.borrow_mut().lap(phase as usize, worker_now());
            }
        };

        let frame_t = time_origin + vsync_ms;
        let period_ms = period.frame(vsync_ms);

        // The scene's own motion: authored clips, texture UV flows and
        // particle emitters, by this frame's real time step (0 on the first).
        let frame_dt_ms = prev_vsync
            .replace(vsync_ms)
            .map_or(0.0, |prev| (vsync_ms - prev).max(0.0));
        if let Err(err) = r.update_animations(frame_dt_ms) {
            tracing::warn!("render thread: update_animations: {err}");
        }
        particles.tick(r, (frame_dt_ms / 1000.0) as f32);
        lap(Phase::Animate);

        // Then the robots' poses, after the clips, so the poses win on the
        // nodes they drive.
        let mut late = false;
        if let Some(l) = link.borrow_mut().as_mut() {
            late = apply_poses(r, l, frame_t);
        }
        lap(Phase::Read);

        feed.borrow()
            .pick(&mut camera.borrow_mut(), frame_t, period_ms);
        let cam = camera.borrow();
        let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
        let mut params =
            CameraParams::perspective(CAMERA_FOV_Y_DEG.to_radians(), CAMERA_NEAR, CAMERA_FAR);
        params.focus_distance = cam.radius;
        let _ = r.set_camera(view, params);
        drop(cam);
        lap(Phase::Camera);

        r.update_transforms();
        lap(Phase::Transforms);
        if let Err(err) = r.render(None) {
            tracing::warn!("render thread: render error: {err}");
        }
        lap(Phase::Render);
        drop(guard);

        frame_count = frame_count.wrapping_add(1);
        let mut c = collector.borrow_mut();
        if frame_count == 3 {
            post_to_main(&RenderMsg::Ready);
            // The scene is on screen: the stats warm up, then count.
            post_report(c.mark_ready());
            poses.restart();
        }
        let input = FrameInput {
            late,
            ..FrameInput::new(vsync_ms, worker_now() - frame_start)
        };
        if let Some(window) = c.end_frame(input, &RendererTimings) {
            let counters = link.borrow().as_ref().map(|l| {
                (
                    l.robots.len() as u32,
                    l.counter(HEADER_FRAMES),
                    l.counter(HEADER_DROPPED),
                )
            });
            let section = poses.section(&mut c, &window, worker_now(), counters);
            post_report(c.report(&window, vec![section]));
        }
        drop(c);
        schedule(&raf_run);
    }));
    schedule(&raf_init);
    std::mem::forget(raf);
    Ok(())
}

/// Asks for the next animation frame for the render loop.
fn schedule(raf: &RafCell) {
    if let Some(cb) = raf.borrow().as_ref() {
        let _ = awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
    }
}

impl PoseLink {
    /// One of the block header's running counters.
    fn counter(&self, index: u32) -> u32 {
        js_sys::Atomics::load(&self.words, index).unwrap_or(0) as u32
    }
}

/// Shows every robot at `t_ms` (absolute): the two samples around it in the
/// robot's ring, blended — positions lerp, rotations nlerp. Past the newest
/// sample the robot holds it (no extrapolation) and the frame is late; before
/// the oldest (the first frames) it shows the oldest.
///
/// Each slot is read under its own seqlock: seq, then the data, then seq
/// again, and a slot that was being written is skipped for this frame (the
/// robot keeps last frame's pose). The writer works several slots ahead of the
/// pair read here, so that is rare, and nothing ever waits.
fn apply_poses(r: &mut awsm_renderer::AwsmRenderer, l: &mut PoseLink, t_ms: f64) -> bool {
    let mut late = false;
    for ring in &mut l.robots {
        let (changed, held) = ring.sample(&l.words, &l.times, t_ms);
        late |= held;
        if changed {
            if let Err(err) =
                awsm_renderer_scene_loader::mujoco::apply_geom_poses(r, &ring.instance, &ring.out)
            {
                tracing::warn!("pose sink ({}): {err}", ring.instance.source.filename);
            }
        }
    }
    late
}

impl RobotRing {
    fn seq_index(&self, k: usize) -> u32 {
        ((self.base + k * self.slot_bytes) / 4) as u32
    }

    /// `(seq, t_ms)` of slot `k` if it holds a stable sample.
    fn stamp(
        &self,
        words: &js_sys::Int32Array,
        times: &js_sys::Float64Array,
        k: usize,
    ) -> Option<(i32, f64)> {
        let seq = js_sys::Atomics::load(words, self.seq_index(k)).ok()?;
        if seq == 0 || seq % 2 != 0 {
            return None; // never written, or being written
        }
        let t = times.get_index(((self.base + k * self.slot_bytes + 8) / 8) as u32);
        (js_sys::Atomics::load(words, self.seq_index(k)).ok()? == seq).then_some((seq, t))
    }

    /// Copies slot `k`'s poses into `into`; false if the slot changed meanwhile.
    fn copy(&self, words: &js_sys::Int32Array, k: usize, seq: i32, into: &mut [f32]) -> bool {
        self.poses[k].copy_to(into);
        js_sys::Atomics::load(words, self.seq_index(k)).ok() == Some(seq)
    }

    /// Puts the poses to show at `t_ms` into `out`. Returns whether `out`
    /// changed (else keep last frame's), and whether `t_ms` was past the
    /// newest sample, which is held (late).
    fn sample(
        &mut self,
        words: &js_sys::Int32Array,
        times: &js_sys::Float64Array,
        t_ms: f64,
    ) -> (bool, bool) {
        let mut older: Option<(usize, i32, f64)> = None;
        let mut newer: Option<(usize, i32, f64)> = None;
        let mut oldest: Option<(usize, i32, f64)> = None;
        for k in 0..RING_SLOTS {
            let Some((seq, t)) = self.stamp(words, times, k) else {
                continue;
            };
            if t <= t_ms {
                if older.is_none_or(|(_, _, o)| t > o) {
                    older = Some((k, seq, t));
                }
            } else if newer.is_none_or(|(_, _, n)| t < n) {
                newer = Some((k, seq, t));
            }
            if oldest.is_none_or(|(_, _, o)| t < o) {
                oldest = Some((k, seq, t));
            }
        }
        let late = older.is_some() && newer.is_none();
        let ((ka, sa, ta), (kb, sb, tb)) = match (older, newer) {
            (Some(o), Some(n)) => (o, n),
            (Some(o), None) => (o, o),
            (None, Some(_)) => match oldest {
                Some(o) => (o, o),
                None => return (false, late),
            },
            (None, None) => return (false, late),
        };
        let alpha = if kb == ka || tb <= ta {
            0.0
        } else {
            ((t_ms - ta) / (tb - ta)).clamp(0.0, 1.0) as f32
        };
        let key = (sa, sb, alpha.to_bits());
        if self.last == Some(key) {
            return (false, late);
        }
        let mut a = std::mem::take(&mut self.a);
        let mut b = std::mem::take(&mut self.b);
        let ok = self.copy(words, ka, sa, &mut a) && (kb == ka || self.copy(words, kb, sb, &mut b));
        if ok {
            if kb == ka {
                self.out.copy_from_slice(&a);
            } else {
                blend(&a, &b, alpha, &mut self.out);
            }
        }
        self.a = a;
        self.b = b;
        if ok {
            self.last = Some(key);
        }
        (ok, late)
    }
}

/// `a` → `b` at fraction `alpha`, per geom: positions lerp, rotations nlerp
/// along the short arc (q and −q are the same rotation).
fn blend(a: &[f32], b: &[f32], alpha: f32, out: &mut [f32]) {
    for ((pa, pb), po) in a
        .as_chunks::<POSE_STRIDE>()
        .0
        .iter()
        .zip(b.as_chunks::<POSE_STRIDE>().0)
        .zip(out.as_chunks_mut::<POSE_STRIDE>().0)
    {
        for k in 0..3 {
            po[k] = pa[k] + (pb[k] - pa[k]) * alpha;
        }
        let dot: f32 = (3..7).map(|k| pa[k] * pb[k]).sum();
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let mut len = 0.0;
        for k in 3..7 {
            po[k] = pa[k] * (1.0 - alpha) + sign * pb[k] * alpha;
            len += po[k] * po[k];
        }
        let len = len.sqrt();
        let len = if len > 0.0 { len } else { 1.0 };
        for v in &mut po[3..7] {
            *v /= len;
        }
    }
}

fn install_onmessage(
    link: Rc<RefCell<Option<PoseLink>>>,
    pending: Rc<RefCell<Option<Vec<awsm_renderer_scene_loader::mujoco::MujocoInstance>>>>,
    feed: Rc<RefCell<CameraFeed>>,
    canvas: web_sys::OffscreenCanvas,
    collector: Rc<RefCell<StatsCollector>>,
    pending_aa: Rc<Cell<Option<AaMode>>>,
) -> Result<(), JsValue> {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    let cb = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        // The pose block is a raw JS object tagged `kind` (it carries a SAB —
        // not serde territory).
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() == Some("pose-block") {
            let Some(instances) = pending.borrow_mut().take() else {
                return;
            };
            let Ok(sab) = js_sys::Reflect::get(&data, &JsValue::from_str("sab")) else {
                return;
            };
            let geoms: Vec<usize> = instances.iter().map(|i| i.geoms.len()).collect();
            let (offsets, _) = ring_offsets(&geoms);
            let robots = instances
                .into_iter()
                .zip(offsets)
                .map(|(instance, base)| {
                    let n = instance.geoms.len() * POSE_STRIDE;
                    let bytes = slot_bytes(instance.geoms.len());
                    let poses = (0..RING_SLOTS)
                        .map(|k| {
                            js_sys::Float32Array::new_with_byte_offset_and_length(
                                &sab,
                                (base + k * bytes + SLOT_POSES) as u32,
                                n as u32,
                            )
                        })
                        .collect();
                    RobotRing {
                        instance,
                        base,
                        slot_bytes: bytes,
                        poses,
                        a: vec![0.0; n],
                        b: vec![0.0; n],
                        out: vec![0.0; n],
                        last: None,
                    }
                })
                .collect();
            *link.borrow_mut() = Some(PoseLink {
                robots,
                words: js_sys::Int32Array::new(&sab),
                times: js_sys::Float64Array::new(&sab),
            });
            post_progress("pose block linked — robots driven");
            return;
        }
        // The settings and stats kit's messages (`{type: …}`).
        if let Some(message) = SettingsToWorker::from_js(&data) {
            match message {
                Ok(SettingsToWorker::AntiAlias { mode }) => pending_aa.set(Some(mode)),
                Ok(SettingsToWorker::Resize { width, height }) => {
                    if width > 0 && height > 0 {
                        canvas.set_width(width);
                        canvas.set_height(height);
                    }
                }
                Err(err) => tracing::warn!("render thread: settings message: {err}"),
            }
            return;
        }
        if let Some(message) = StatsToWorker::from_js(&data) {
            match message {
                Ok(message) => collector.borrow_mut().handle(message),
                Err(err) => tracing::warn!("render thread: stats message: {err}"),
            }
            return;
        }
        if let Ok(msg) = serde_wasm_bindgen::from_value::<CameraMsg>(data) {
            feed.borrow_mut().push(msg);
        }
    });
    scope.set_onmessage(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
    Ok(())
}

async fn fetch_scene(url: &str) -> Result<awsm_renderer_scene::Scene, String> {
    let text = gloo_net::http::Request::get(url)
        .cache(web_sys::RequestCache::NoCache)
        .send()
        .await
        .map_err(|e| format!("fetch: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read: {e}"))?;
    awsm_renderer_scene::project_dir::scene_from_toml(&text).map_err(|e| format!("parse: {e}"))
}

/// Post a human-readable load-progress line to main (the loading screen).
fn post_progress(message: &str) {
    post_to_main(&RenderMsg::Progress {
        message: message.to_string(),
    });
}

/// Serialize a [`RenderMsg`] and post it to the main thread.
fn post_to_main(msg: &RenderMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    match serde_wasm_bindgen::to_value(msg) {
        Ok(v) => {
            let _ = scope.post_message(&v);
        }
        Err(e) => tracing::error!("render thread: serialize RenderMsg: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_lerps_positions_and_takes_the_short_arc() {
        // One geom: x 0 → 2, rotation identity → identity written as -q.
        let a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let b = [2.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let mut out = [0.0; 7];
        blend(&a, &b, 0.25, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-6);
        // -q is the same rotation: the blend stays at identity, unit length.
        assert!((out[3] - 1.0).abs() < 1e-6);
        assert!(out[4..].iter().all(|v| v.abs() < 1e-6));
    }
}
