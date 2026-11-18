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

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html};
use futures_signals::signal::{Mutable, SignalExt};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, MessageEvent, Worker, WorkerOptions, WorkerType};

use crate::bootstrap::spawn_shared_worker_transfer;
use crate::camera::{CameraMsg, OrbitCamera};
use crate::protocol::{pose_block_bytes, ProfileMsg, RenderMsg};
use crate::stats::Stats;

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
                if let Err(e) = setup(canvas, status.clone(), source.clone(), errors.clone()) {
                    status.set(format!("setup error: {e:?}"));
                    tracing::error!("main thread setup: {e:?}");
                }
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

fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    source: Mutable<String>,
    errors: Mutable<String>,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    // Take keyboard focus (the Settings modal closes on Escape). The page is
    // usually embedded in an iframe on awsm.fun; a click on the canvas focuses
    // it too, since no pointer handler here cancels the press.
    let _ = window.focus();
    maybe_reset_settings(&window);
    let stream_url = query(&window, "stream");

    // The Stats chip + panel (see `crate::stats`), fed by the render worker.
    let stats = Stats::install(&window, STORAGE_PREFIX, stream_url.is_some());

    // Backing store at the chosen render resolution (a stored Settings choice,
    // else native capped at a pixel budget — see `resolution`). Must be sized
    // BEFORE the transfer.
    let resolution = crate::resolution::Resolution::new(&window, &canvas, "awsm_isaac_res_scale");
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
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    let render_cell: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));
    let started = Rc::new(Cell::new(false));
    let on_render_msg = Closure::<dyn FnMut(MessageEvent)>::new(clone!(
        status, render_cell, started, pose, stats => move |e: MessageEvent| {
            match serde_wasm_bindgen::from_value::<RenderMsg>(e.data()) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::SceneReady { instances }) => {
                    if started.replace(true) {
                        return;
                    }
                    // The worker listens from now on: tell it which Stats
                    // groups are expanded (later changes follow, below).
                    if let Some(render) = render_cell.borrow().as_ref() {
                        post_profile(render, &stats);
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
                    stats.mark_ready();
                    status.set("running — drag orbits, wheel zooms".into());
                }
                Ok(RenderMsg::Stats(report)) => stats.window(report),
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
    wasm_bindgen_futures::spawn_local(stats.profile_signal().for_each(
        clone!(render, started, stats => move |_| {
            if started.get() {
                post_profile(&render, &stats);
            }
            async {}
        }),
    ));

    // Layout changes → ResizeMsg, plus the Settings modal (resolution slider).
    resolution.install(&window, &canvas, render.clone())?;
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
    stats: Rc<Stats>,
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
    let time_origin = window.performance().map(|p| p.time_origin()).unwrap_or(0.0);
    type Cb = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let cb: Cb = Rc::new(RefCell::new(None));
    let again = cb.clone();
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        // This is the page's per-frame work (the Stats panel's page group).
        stats.time_camera(|| {
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
        });
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

/// Tell the render worker which Stats groups are expanded.
fn post_profile(render: &Worker, stats: &Stats) {
    if let Ok(v) = serde_wasm_bindgen::to_value(&ProfileMsg::Set(stats.profile())) {
        let _ = render.post_message(&v);
    }
}

/// `?reset` in the URL forgets every stored setting of this app (the panel,
/// its groups, the render resolution), so the page boots with the defaults.
/// Runs before anything reads one.
fn maybe_reset_settings(window: &web_sys::Window) {
    let reset = query_has(window, "reset");
    let Some(ls) = reset
        .then(|| window.local_storage().ok().flatten())
        .flatten()
    else {
        return;
    };
    let keys: Vec<String> = (0..ls.length().unwrap_or(0))
        .filter_map(|i| ls.key(i).ok().flatten())
        .filter(|k| k.starts_with(STORAGE_PREFIX))
        .collect();
    for key in keys {
        let _ = ls.remove_item(&key);
    }
    loading_log("?reset — stored settings cleared, booting with defaults");
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

/// Whether the page's query string has `key` (with or without a value).
fn query_has(window: &web_sys::Window, key: &str) -> bool {
    window
        .location()
        .search()
        .ok()
        .and_then(|s| web_sys::UrlSearchParams::new_with_str(&s).ok())
        .is_some_and(|p| p.has(key))
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
