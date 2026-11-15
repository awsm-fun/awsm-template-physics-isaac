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
//! 2. each frame, copies the latest stable snapshot out of the block (seqlock —
//!    see [`crate::protocol`]) and hands each robot's span to
//!    `scene_loader::mujoco::apply_geom_poses`.
//!
//! The pose worker already refused any capture or stream frame whose
//! fingerprint does not match its robot, so what arrives here is bound.

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

use crate::protocol::{CameraMsg, InstanceInfo, RenderMsg, ResizeMsg, POSE_HEADER, POSE_STRIDE};

/// The boxed RAF callback, self-referenced so the render loop can reschedule
/// itself each frame (and stays alive for the worker's lifetime).
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Lens + clip planes for the one camera this app has.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.05;
const CAMERA_FAR: f32 = 400.0;

/// A minimal mouse-driven orbit camera: drag to orbit, wheel to dolly.
struct OrbitCamera {
    look_at: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

impl OrbitCamera {
    const SENSITIVITY: f32 = 0.008;
    const PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const PITCH_MIN: f32 = 0.02;
    const MIN_RADIUS: f32 = 0.8;
    const MAX_RADIUS: f32 = 20.0;

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * Self::SENSITIVITY;
        self.pitch = (self.pitch - dy * Self::SENSITIVITY).clamp(Self::PITCH_MIN, Self::PITCH_MAX);
    }

    fn zoom(&mut self, dy: f32) {
        self.radius = (self.radius * (1.0 + dy * 0.001)).clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }

    fn eye(&self) -> Vec3 {
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

/// The live link between the pose block and the scene's robots.
struct PoseLink {
    /// Each robot's loader-resolved instance and its span in the block
    /// (offset in f32 slots from the start of the pose span).
    robots: Vec<(awsm_renderer_scene_loader::mujoco::MujocoInstance, usize)>,
    /// i32 view over the whole block (seq at 0).
    header: js_sys::Int32Array,
    /// f32 view over the pose span (`POSE_HEADER..`).
    poses: js_sys::Float32Array,
    /// Copy target for one stable snapshot.
    scratch: Vec<f32>,
    /// Last successfully applied seq — skip work when nothing new arrived.
    last_seq: i32,
}

/// Worker entry: unpack the transferred `OffscreenCanvas` + page origin, build
/// the WebGPU device, and kick off the async load + render.
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
            // otherwise pair it with the new JS. Bump with awsm-renderer.
            format!("{base}/workers/basis-worker.js?v=0.42.0"),
            format!("{base}/vendor/basis/basis_transcoder.js?v=0.42.0"),
        ));
    }
    let canvas_handle = canvas.clone();
    post_progress("render worker: requesting WebGPU device…");
    let gpu =
        navigator_gpu().ok_or_else(|| JsValue::from_str("render worker: no navigator.gpu"))?;
    let gpu_builder = AwsmRendererWebGpuBuilder::new_with_offscreen_canvas(gpu, canvas)
        .with_device_request_limits(DeviceRequestLimits::max_all());

    wasm_bindgen_futures::spawn_local(async move {
        if let Err(err) = run(gpu_builder, canvas_handle, origin).await {
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
) -> Result<(), JsValue> {
    use awsm_renderer::camera::CameraParams;
    use awsm_renderer::AwsmRendererBuilder;

    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(awsm_renderer::anti_alias::AntiAliasing {
            msaa_sample_count: Some(4),
            smaa: false,
            mipmap: true,
        })
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
    post_progress("render worker: WebGPU device + renderer ready");

    // ── Warm-up + scene fetch, CONCURRENTLY ─────────────────────────────────
    let bundle_base = format!("{}/bundle", origin.trim_end_matches('/'));
    let scene_url = format!("{bundle_base}/scene.toml");
    post_progress("compiling core render pipelines + fetching scene… (first visit can take a while — cached after)");
    let (compiled, scene) =
        futures::join!(renderer.ensure_config_pipelines(), fetch_scene(&scene_url));
    let compiled =
        compiled.map_err(|e| JsValue::from_str(&format!("ensure_config_pipelines: {e}")))?;
    let scene = scene.map_err(|e| JsValue::from_str(&format!("load scene {scene_url}: {e}")))?;
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
    let camera = Rc::new(RefCell::new(OrbitCamera {
        look_at: Vec3::new(-0.1, 0.45, 0.0),
        radius: 2.7,
        yaw: 0.45,
        pitch: 0.22,
    }));
    #[allow(clippy::arc_with_non_send_sync)]
    let cell = Rc::new(RefCell::new(renderer));
    let link: Rc<RefCell<Option<PoseLink>>> = Rc::new(RefCell::new(None));
    let pending = Rc::new(RefCell::new(Some(instances)));

    install_onmessage(link.clone(), pending, camera.clone(), canvas.clone())?;
    // Only now is the worker listening — main allocates the pose block from
    // this message.
    post_to_main(&RenderMsg::SceneReady { instances: infos });

    // ── Render loop ─────────────────────────────────────────────────────────
    #[allow(clippy::arc_with_non_send_sync)]
    let raf: RafCell = Rc::new(RefCell::new(None));
    let raf_init = raf.clone();
    let raf_run = raf.clone();
    let mut frame_count: u32 = 0;

    *raf_init.borrow_mut() = Some(Closure::new(move |_vsync_ms: f64| {
        let mut r = cell.borrow_mut();

        if let Some(l) = link.borrow_mut().as_mut() {
            apply_poses(&mut r, l);
        }

        let cam = camera.borrow();
        let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
        let mut params =
            CameraParams::perspective(CAMERA_FOV_Y_DEG.to_radians(), CAMERA_NEAR, CAMERA_FAR);
        params.focus_distance = cam.radius;
        let _ = r.set_camera(view, params);
        drop(cam);

        r.update_transforms();
        if let Err(err) = r.render(None) {
            tracing::warn!("render thread: render error: {err}");
        }
        drop(r);

        frame_count = frame_count.wrapping_add(1);
        if frame_count == 3 {
            post_to_main(&RenderMsg::Ready);
        }
        if let Some(cb) = raf_run.borrow().as_ref() {
            let _ = awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }));
    if let Some(cb) = raf_init.borrow().as_ref() {
        awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref())?;
    }
    std::mem::forget(raf);
    Ok(())
}

/// Copy the latest stable snapshot out of the seqlock'd block and hand each
/// robot its span. On a torn read (seq odd / changed under us) keep last
/// frame's poses — the next frame catches up.
///
/// The entire per-frame cost of the integration: a seqlock read and one sink
/// call per robot. Every geom-to-node decision was made at import time.
fn apply_poses(r: &mut awsm_renderer::AwsmRenderer, l: &mut PoseLink) {
    let seq0 = js_sys::Atomics::load(&l.header, 0).unwrap_or(0);
    if seq0 == l.last_seq || seq0 % 2 != 0 || seq0 == 0 {
        return;
    }
    l.poses.copy_to(&mut l.scratch[..]);
    let seq1 = js_sys::Atomics::load(&l.header, 0).unwrap_or(-1);
    if seq0 != seq1 {
        return; // torn — the writer was mid-publish
    }
    l.last_seq = seq0;
    for (instance, offset) in &l.robots {
        let span = &l.scratch[*offset..*offset + instance.geoms.len() * POSE_STRIDE];
        if let Err(err) = awsm_renderer_scene_loader::mujoco::apply_geom_poses(r, instance, span) {
            tracing::warn!("pose sink ({}): {err}", instance.source.filename);
        }
    }
}

fn install_onmessage(
    link: Rc<RefCell<Option<PoseLink>>>,
    pending: Rc<RefCell<Option<Vec<awsm_renderer_scene_loader::mujoco::MujocoInstance>>>>,
    camera: Rc<RefCell<OrbitCamera>>,
    canvas: web_sys::OffscreenCanvas,
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
            let total: usize = instances.iter().map(|i| i.geoms.len()).sum();
            let mut offset = 0;
            let robots = instances
                .into_iter()
                .map(|i| {
                    let at = offset;
                    offset += i.geoms.len() * POSE_STRIDE;
                    (i, at)
                })
                .collect();
            *link.borrow_mut() = Some(PoseLink {
                robots,
                header: js_sys::Int32Array::new(&sab),
                poses: js_sys::Float32Array::new(&sab).subarray(
                    POSE_HEADER as u32,
                    (POSE_HEADER + total * POSE_STRIDE) as u32,
                ),
                scratch: vec![0.0; total * POSE_STRIDE],
                last_seq: 0,
            });
            post_progress("pose block linked — robots driven");
            return;
        }
        if let Ok(ResizeMsg::Canvas { width, height }) =
            serde_wasm_bindgen::from_value::<ResizeMsg>(data.clone())
        {
            if width > 0 && height > 0 {
                canvas.set_width(width);
                canvas.set_height(height);
            }
            return;
        }
        match serde_wasm_bindgen::from_value::<CameraMsg>(data) {
            Ok(CameraMsg::Orbit { dx, dy }) => camera.borrow_mut().orbit(dx, dy),
            Ok(CameraMsg::Zoom { dy }) => camera.borrow_mut().zoom(dy),
            Err(_) => {}
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
