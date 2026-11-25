//! Main thread — owns the DOM (built with Dominator) and orchestrates the two
//! workers.
//!
//! Flow:
//! 1. Build the canvas + HUD, transfer the canvas to an `OffscreenCanvas`, and
//!    spawn the **render** worker (a Rust thread sharing this wasm module +
//!    memory). It loads the bundle and resolves the scene's sim instances.
//! 2. Spawn the **pose** worker (`web/workers/pose-worker.js`) and tell it its
//!    source: replay the captures (default) or `?stream=ws://…`.
//! 3. When the render worker reports `SceneReady { instances }`, allocate the
//!    `SharedArrayBuffer` pose block sized for exactly those robots, and hand
//!    it to both workers. From then on poses flow pose worker → render worker
//!    through shared memory only; main relays resize and the pose worker's
//!    status line, and owns the orbit camera: once per frame, if it moved,
//!    main posts its state to the render worker (see `crate::camera`).
//!
//! Settings (bottom right: the render resolution and the anti-aliasing) and
//! the Stats panel are awsm-renderer-kit's panels: `SettingsPanel` and
//! `StatsPanel` here, the kit's messages to and from the render worker (see
//! [`WorkerLink`] and `crate::stats`).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::Mutable;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

use awsm_renderer_settings::{
    probe_gpu, AaMode, SettingsConfig, SettingsHooks, SettingsOptions, SettingsPanel,
    SettingsStore, SettingsToWorker, SettingsUi,
};
use awsm_renderer_stats::{StatsCollector, StatsHooks, StatsPanel, StatsToPage, StatsToWorker};

use crate::about::{About, Tab};
use crate::bootstrap::spawn_shared_worker_transfer;
use crate::camera::{CameraMsg, OrbitCamera};
use crate::protocol::{pose_block_bytes, RenderMsg};
use crate::stats as app_stats;

/// Every `localStorage` key this app writes starts with this (the CDN origin is
/// shared with other experiments).
const STORAGE_PREFIX: &str = "awsm_isaac_";

/// Build + mount the DOM, then start the worker pipeline.
pub fn start() -> Result<(), JsValue> {
    let status = Mutable::new("booting…".to_string());
    let source = Mutable::new(String::new());
    let errors = Mutable::new(String::new());

    let app = html!("div", {
        .child(html!("canvas" => HtmlCanvasElement, {
            .class("canvas")
            .after_inserted(clone!(status, source, errors => move |canvas| {
                wasm_bindgen_futures::spawn_local(async move {
                    if let Err(e) = setup(canvas, status.clone(), source, errors).await {
                        status.set(format!("setup error: {e:?}"));
                        tracing::error!("main thread setup: {e:?}");
                    }
                });
            }))
        }))
        .child(html!("div", {
            .class("hud")
            .text("physics-isaac — Isaac Sim robots in awsm-renderer\ndrag: orbit · wheel: zoom\nrobots: Franka Panda (Apache-2.0) · ANYmal-D (BSD-3, © ANYbotics AG) — see ASSET-LICENSES.md\n")
            .child(html!("div", {
                .text_signal(source.signal_cloned())
            }))
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
            .child(html!("div", {
                .class("hud-error")
                .text_signal(errors.signal_cloned())
            }))
        }))
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

/// This app's Settings: every anti-aliasing mode, TAA by default on a desktop
/// and Off on a touch device or a software GPU, and the shared resolution
/// policy (awsm-renderer-settings).
fn settings_config() -> SettingsConfig {
    SettingsConfig::default()
}

/// The Settings modal's words and look: this template's close button and
/// hint.
fn settings_ui() -> SettingsUi {
    SettingsUi {
        close_class: "settings-close".into(),
        hint: Some("Lower the resolution if your framerate is low.".into()),
        ..SettingsUi::default()
    }
}

async fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    source: Mutable<String>,
    errors: Mutable<String>,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| JsValue::from_str("no document"))?;
    let body = dominator::body();
    // Take keyboard focus (the modals close on Escape). The page is
    // usually embedded in an iframe on awsm.fun; a click on the canvas focuses
    // it too, since no pointer handler here cancels the press.
    let _ = window.focus();
    // `?reset` forgets the stored settings (the resolution, the
    // anti-aliasing, the Stats panel and its groups), so the page boots with
    // the defaults. Before anything reads one.
    let store = SettingsStore::new(STORAGE_PREFIX);
    if store.reset_if_asked() {
        loading_log("?reset — stored settings cleared, booting with defaults");
    }
    let stream_url = query(&window, "stream");
    let stream = stream_url.is_some();
    // The kit's messages to the render worker; held until it listens.
    let link = Rc::new(WorkerLink::default());

    // The Stats chip + panel (awsm-renderer-stats), fed by the render worker.
    // Its `what do these mean?` opens About on the Stats tab. Mounted between
    // the Stats and Settings buttons so the Tab order runs left to right.
    let about = About::new();
    let stats_panel: web_sys::HtmlElement = document.create_element("div")?.unchecked_into();
    stats_panel.set_class_name("stats");
    let stats_button: web_sys::HtmlElement = document.create_element("button")?.unchecked_into();
    stats_button.set_class_name("stats-btn");
    stats_button.set_attribute("type", "button")?;
    stats_button.set_text_content(Some("Stats"));
    body.append_child(&stats_panel)?;
    body.append_child(&stats_button)?;
    let stats = StatsPanel::mount(
        STORAGE_PREFIX,
        &stats_panel,
        Some(&stats_button),
        StatsHooks {
            post: Box::new(clone!(link => move |message| link.post_stats(message))),
            on_help: Some(Box::new(clone!(about => move || about.open(Tab::Stats)))),
            page_part: app_stats::PAGE_PART.to_string(),
            app_sections: None,
            section_rows: None,
            shown: None,
        },
    );
    // "starting…" in every group until the render worker reports.
    stats.handle(&StatsToPage::Stats(
        StatsCollector::new(app_stats::config(stream)).placeholder(),
    ));
    about.install();

    // The Settings button (bottom right) and its modal (awsm-renderer-settings).
    let settings_button = document.create_element("button")?;
    settings_button.set_class_name("settings-btn");
    settings_button.set_attribute("type", "button")?;
    settings_button.set_text_content(Some("Settings"));
    body.append_child(&settings_button)?;
    let settings_root = document.create_element("div")?;
    body.append_child(&settings_root)?;
    // A software GPU starts at a lower resolution and without anti-aliasing,
    // so the adapter is asked first (a separate, lightweight request).
    let gpu = probe_gpu().await;
    tracing::info!(
        "gpu: fallback adapter {}, largest 2D texture {}",
        gpu.is_fallback,
        gpu.max_texture_dim
    );
    let settings = SettingsPanel::new(
        &settings_root,
        &canvas,
        store,
        &SettingsOptions::new(&settings_config(), Some(gpu)),
        settings_ui(),
        loading_log,
    );

    // Backing store at the chosen render resolution (a stored Settings choice,
    // else native capped at a pixel budget). Must be sized BEFORE the
    // transfer.
    let (width, height) = settings.backing();
    canvas.set_width(width);
    canvas.set_height(height);
    let offscreen = canvas.transfer_control_to_offscreen()?;

    let base = page_base(&window);
    let app_base = app_base(&window);

    // ── Pose worker (plain JS module worker) ────────────────────────────────
    loading_log("spawning pose worker…");
    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let ab = app_base.trim_end_matches('/');
    let pose = Worker::new_with_options(&format!("{ab}/workers/pose-worker.js"), &opts)?;
    let on_pose_msg =
        Closure::<dyn FnMut(MessageEvent)>::new(clone!(errors, source => move |e: MessageEvent| {
            let data = e.data();
            let field = |k: &str| {
                js_sys::Reflect::get(&data, &JsValue::from_str(k))
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_default()
            };
            match field("kind").as_str() {
                "progress" => loading_log(&field("message")),
                // The pose worker owns the "what is driving the robots" line — it
                // is the only side that knows (and it changes: a stream connects,
                // drops, reconnects).
                "source" => source.set(field("message")),
                // Errors get their own line: the status and source lines keep
                // changing, and a robot that is NOT driven must stay visible.
                "error" => {
                    let m = field("message");
                    loading_log(&format!("POSE SOURCE ERROR: {m}"));
                    errors.replace_with(|all| {
                        if all.is_empty() { m } else { format!("{all}\n{m}") }
                    });
                }
                _ => {}
            }
        }));
    pose.set_onmessage(Some(on_pose_msg.as_ref().unchecked_ref()));
    on_pose_msg.forget();
    let init = js_sys::Object::new();
    set(&init, "kind", &JsValue::from_str("init"));
    set(
        &init,
        "captures_base",
        &JsValue::from_str(&format!("{}/captures", base.trim_end_matches('/'))),
    );
    if let Some(url) = &stream_url {
        set(&init, "stream", &JsValue::from_str(url));
    }
    pose.post_message(&init)?;

    // ── Render worker ───────────────────────────────────────────────────────
    let payload = js_sys::Object::new();
    set(&payload, "canvas", &offscreen);
    set(&payload, "origin", &JsValue::from_str(&base));
    set(&payload, "app_base", &JsValue::from_str(&app_base));
    // The renderer is built with the chosen mode (and the stats label follows
    // the pose source).
    set(&payload, "aa", &JsValue::from_str(settings.aa().id()));
    set(&payload, "stream", &JsValue::from_bool(stream));
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    let render_cell: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));
    let started = Rc::new(Cell::new(false));
    let on_render_msg = Closure::<dyn FnMut(MessageEvent)>::new(clone!(
        status, render_cell, started, pose, stats, settings, link => move |e: MessageEvent| {
            let data = e.data();
            // The kit's messages first: the compile overlay, the Stats report.
            if settings.handle_js(&data) || stats.handle_js(&data) {
                return;
            }
            match serde_wasm_bindgen::from_value::<RenderMsg>(data) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::SceneReady { instances }) => {
                    if started.replace(true) {
                        return;
                    }
                    // The worker listens from now on: send it what the panels
                    // said meanwhile (which Stats groups are open, the
                    // canvas size, a Settings change).
                    if let Some(render) = render_cell.borrow().as_ref() {
                        link.listening(render.clone());
                    }
                    let geoms: Vec<usize> = instances.iter().map(|i| i.geoms).collect();
                    loading_log(&format!(
                        "scene ready — {} robot(s): {}",
                        instances.len(),
                        instances.iter().map(|i| format!("{} ({} geoms)", i.model, i.geoms)).collect::<Vec<_>>().join(", ")
                    ));
                    // Sized for exactly the scene's robots, in its instance
                    // order — the layout both workers agree on.
                    let sab = js_sys::SharedArrayBuffer::new(pose_block_bytes(&geoms) as u32);
                    let start = js_sys::Object::new();
                    set(&start, "kind", &JsValue::from_str("start"));
                    set(&start, "sab", &sab);
                    set(&start, "instances", &serde_wasm_bindgen::to_value(&instances).unwrap_or(JsValue::NULL));
                    let _ = pose.post_message(&start);
                    if let Some(render) = render_cell.borrow().as_ref() {
                        let block = js_sys::Object::new();
                        set(&block, "kind", &JsValue::from_str("pose-block"));
                        set(&block, "sab", &sab);
                        let _ = render.post_message(&block);
                    }
                }
                Ok(RenderMsg::Ready) => {
                    loading_log("first frames rendered — ready");
                    loading_done();
                    status.set("running — drag orbits, wheel zooms".into());
                }
                Ok(RenderMsg::Error { message }) => {
                    loading_log(&format!("ERROR: {message}"));
                    status.set(format!("render error: {message}"));
                }
                Err(_) => { /* not a RenderMsg (init-error blob) — ignore */ }
            }
        }
    ));

    loading_log("spawning render worker…");
    let render = spawn_shared_worker_transfer(
        "render",
        &payload,
        &transfer,
        on_render_msg.as_ref().unchecked_ref(),
    )?;
    on_render_msg.forget();
    *render_cell.borrow_mut() = Some(render.clone());

    // Layout changes and the Settings modal → `resize` / `anti-alias`.
    settings.start(
        &canvas,
        SettingsHooks {
            on_resize: Box::new(clone!(link => move |width, height| {
                link.post_settings(SettingsToWorker::Resize { width, height });
            })),
            on_anti_alias: Box::new(clone!(link => move |mode| {
                link.post_settings(SettingsToWorker::AntiAlias { mode });
            })),
            open_button: Some(settings_button),
            extra_rows: Vec::new(),
        },
    );
    install_pointer(&canvas, render, stats)?;

    status.set("loading scene…".into());
    Ok(())
}

/// Pointer-drag orbits, wheel zooms. The camera lives here, clamps and all;
/// each main-thread frame in which it changed, its state goes to the render
/// worker stamped with that frame's time (see `crate::camera` for why it is
/// once per frame and not once per event).
fn install_pointer(
    canvas: &HtmlCanvasElement,
    render: Worker,
    stats: StatsPanel,
) -> Result<(), JsValue> {
    let camera = Rc::new(RefCell::new(OrbitCamera::default()));
    let dirty = Rc::new(Cell::new(false));
    let dragging = Rc::new(Cell::new(false));
    let last: Rc<Cell<(f32, f32)>> = Rc::new(Cell::new((0.0, 0.0)));

    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, last => move |e: web_sys::PointerEvent| {
            dragging.set(true);
            last.set((e.client_x() as f32, e.client_y() as f32));
        }),
    );
    canvas.add_event_listener_with_callback("pointerdown", down.as_ref().unchecked_ref())?;
    down.forget();

    let up = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging => move |_e: web_sys::PointerEvent| {
            dragging.set(false);
        }),
    );
    canvas.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref())?;
    canvas.add_event_listener_with_callback("pointercancel", up.as_ref().unchecked_ref())?;
    up.forget();

    let mv = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, last, camera, dirty => move |e: web_sys::PointerEvent| {
            if !dragging.get() {
                return;
            }
            let (lx, ly) = last.get();
            let (x, y) = (e.client_x() as f32, e.client_y() as f32);
            last.set((x, y));
            camera.borrow_mut().orbit(x - lx, y - ly);
            dirty.set(true);
        }),
    );
    canvas.add_event_listener_with_callback("pointermove", mv.as_ref().unchecked_ref())?;
    mv.forget();

    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(
        clone!(camera, dirty => move |e: web_sys::WheelEvent| {
            e.prevent_default();
            camera.borrow_mut().zoom(e.delta_y() as f32);
            dirty.set(true);
        }),
    );
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();

    // The per-frame post. rAF timestamps are relative to this thread's time
    // origin; the render worker compares them with its own frame times, so
    // both are made absolute.
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let performance = window.performance();
    let time_origin = performance.as_ref().map_or(0.0, |p| p.time_origin());
    type Cb = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let cb: Cb = Rc::new(RefCell::new(None));
    let again = cb.clone();
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        // This is the page's per-frame work (the Stats panel's page group).
        let t0 = performance.as_ref().map_or(0.0, |p| p.now());
        {
            if dirty.replace(false) {
                let cam = camera.borrow();
                let msg = CameraMsg::Pose {
                    t_ms: time_origin + t,
                    yaw: cam.yaw,
                    pitch: cam.pitch,
                    radius: cam.radius,
                };
                if let Ok(v) = serde_wasm_bindgen::to_value(&msg) {
                    let _ = render.post_message(&v);
                }
            }
        }
        let t1 = performance.as_ref().map_or(0.0, |p| p.now());
        stats.page_frame(t1 - t0, None, true);
        if let (Some(w), Some(f)) = (web_sys::window(), again.borrow().as_ref()) {
            let _ = w.request_animation_frame(f.as_ref().unchecked_ref());
        }
    }));
    if let Some(f) = cb.borrow().as_ref() {
        window.request_animation_frame(f.as_ref().unchecked_ref())?;
    }
    std::mem::forget(cb);
    Ok(())
}

/// The kit's messages from the panels to the render worker. The worker
/// listens only once its scene is loaded (`SceneReady`); until then the
/// latest of each kind waits here (the panel's last `stats-view`, the latest
/// canvas size, a Settings change), and the page's once-a-second samples are
/// dropped.
#[derive(Default)]
struct WorkerLink {
    worker: RefCell<Option<Worker>>,
    view: RefCell<Option<StatsToWorker>>,
    size: Cell<Option<(u32, u32)>>,
    aa: Cell<Option<AaMode>>,
}

impl WorkerLink {
    fn post_settings(&self, message: SettingsToWorker) {
        if let Some(worker) = self.worker.borrow().as_ref() {
            post_kit(worker, message.to_js());
            return;
        }
        match message {
            SettingsToWorker::Resize { width, height } => self.size.set(Some((width, height))),
            SettingsToWorker::AntiAlias { mode } => self.aa.set(Some(mode)),
        }
    }

    fn post_stats(&self, message: &StatsToWorker) {
        if let Some(worker) = self.worker.borrow().as_ref() {
            post_kit(worker, message.to_js());
        } else if matches!(message, StatsToWorker::StatsView { .. }) {
            *self.view.borrow_mut() = Some(message.clone());
        }
    }

    /// The render worker listens from now on: send what waited.
    fn listening(&self, worker: Worker) {
        *self.worker.borrow_mut() = Some(worker);
        let view = self.view.borrow_mut().take();
        if let Some(view) = view {
            self.post_stats(&view);
        }
        if let Some((width, height)) = self.size.take() {
            self.post_settings(SettingsToWorker::Resize { width, height });
        }
        if let Some(mode) = self.aa.take() {
            self.post_settings(SettingsToWorker::AntiAlias { mode });
        }
    }
}

fn post_kit(worker: &Worker, message: Result<JsValue, String>) {
    match message {
        Ok(value) => {
            let _ = worker.post_message(&value);
        }
        Err(err) => tracing::error!("kit message: {err}"),
    }
}

/// Append a line to the `#loading-log` panel (the pre-Ready progress feed).
pub fn loading_log(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(log) = document.get_element_by_id("loading-log") else {
        // Loading screen already dismissed — dev-console only.
        tracing::info!("{message}");
        return;
    };
    if let Ok(div) = document.create_element("div") {
        div.set_text_content(Some(message));
        let _ = log.append_child(&div);
    }
    tracing::info!("loading: {message}");
}

/// Fade out + disable the loading overlay (first frames are on screen).
fn loading_done() {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("loading"))
    {
        let _ = el.class_list().add_1("done");
    }
}

/// Paint a fatal boot error onto the loading screen (no renderer will come up).
pub fn fatal(message: &str) {
    loading_log(message);
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("loading-log"))
    {
        let _ = el.set_attribute("style", "color:#ff6a5e");
    }
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

/// A query-string parameter of the page, if present and non-empty.
fn query(window: &web_sys::Window, key: &str) -> Option<String> {
    let search = window.location().search().ok()?;
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get(key).filter(|v| !v.is_empty())
}

/// The *media* base URL: `?media=` query param → compile-time `MEDIA_BASE`
/// (set by `task dev` to the side media server) → the page's own directory.
fn page_base(window: &web_sys::Window) -> String {
    if let Some(media) = query(window, "media") {
        return media.trim_end_matches('/').to_string();
    }
    if let Some(base) = option_env!("MEDIA_BASE") {
        return base.trim_end_matches('/').to_string();
    }
    app_base(window)
}

/// The *app* base URL: the directory this page is served from (where Trunk laid
/// down `workers/` + `vendor/`). Survives subpath deploys.
fn app_base(window: &web_sys::Window) -> String {
    let location = window.location();
    let origin = location.origin().unwrap_or_default();
    let path = location.pathname().unwrap_or_default();
    let dir = match path.rfind('/') {
        Some(idx) => &path[..idx],
        None => "",
    };
    format!("{origin}{dir}")
}
