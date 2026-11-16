// Pose worker — the template's pose SOURCE. Publishes every robot's geom world
// poses into the SharedArrayBuffer pose block (layout: src/protocol.rs) as
// time-stamped samples; the render worker interpolates them at its own frame
// time, so how evenly this worker's timers or the network run never shows.
//
// Two sources, indistinguishable to the rest of the app:
//
//   replay (default)    loops media/captures/<model>.capture.json — a scripted
//                       joint sweep recorded offline by forward kinematics
//                       over the USD joints (awsm-renderer-isaac-record).
//                       Joint animation, not physics. Each capture frame is
//                       published stamped with when it is due on the loop's
//                       timeline, a little ahead of time.
//   stream (?stream=)   binary frames from a WebSocket: 32 raw bytes of the
//                       robot's SHA-256 fingerprint, optionally a u32 step +
//                       f32 dt, then 7 × geom_count f32 little-endian
//                       [px,py,pz,qw,qx,qy,qz] (see src/protocol.rs). Fed by an
//                       Isaac Lab harness or harness/replay_stream.py. Frames
//                       are stamped step·dt mapped onto the page clock, plus a
//                       short playout delay.
//
// Protocol with main:
//   main → here  {kind:"init", captures_base, stream?}
//   main → here  {kind:"start", sab, instances:[{model, sha256, geoms}]}
//   here → main  {kind:"progress", message} | {kind:"source", message}
//                {kind:"error", message}
//
// Every capture and every stream frame is checked against its robot's
// fingerprint and geom count before a single pose is written: a mismatch is a
// loud error or a counted drop, never a robot driven by another robot's poses.

// Mirrors of src/protocol.rs.
const POSE_STRIDE = 7;
const HEADER_BYTES = 16;
const BLOCK_LAYOUT = 2;
const RING_SLOTS = 8;
const SLOT_POSES = 16;
const SOURCE_REPLAY = 1;
const SOURCE_STREAM = 2;
const slotBytes = (geoms) => Math.ceil((SLOT_POSES + geoms * POSE_STRIDE * 4) / 8) * 8;

// Replay: how far ahead of now capture frames are published (ms). The render
// worker needs the sample just after its frame time, so a little lead means it
// is always there; the cap keeps the writer clear of the slots being read.
const REPLAY_LEAD_MS = 100;
const REPLAY_TICK_MS = 10;
// Stream: playout delay in steps, the window (in frames) of the smallest
// (arrival − model time) that maps the producer's clock onto ours, and how many
// arrival intervals estimate dt for a plain-form (stepless) stream.
const PLAYOUT_STEPS = 2.5;
const OFFSET_WINDOW = 64;
const DT_WINDOW = 32;

// The page clock every thread shares: absolute ms.
const nowAbs = () => performance.timeOrigin + performance.now();

let config = null;
let header = null; // Int32Array over the header [layout, frames, source, dropped]
let robots = [];   // {model, sha256, geoms, ring: {seq: [Int32Array], t: [Float64Array], poses: [Float32Array]}, next}

const post = (obj) => self.postMessage(obj);
const progress = (message) => post({ kind: "progress", message: `poses: ${message}` });
const setSource = (message) => post({ kind: "source", message });
const fail = (message) => {
    console.error("pose worker:", message);
    post({ kind: "error", message });
};

self.onmessage = async (e) => {
    const d = e.data;
    if (!d || !d.kind) return;
    try {
        if (d.kind === "init") {
            config = d;
        } else if (d.kind === "start") {
            start(d.sab, d.instances);
        }
    } catch (err) {
        fail((err && err.message) ? err.message : String(err));
    }
};

function start(sab, instances) {
    header = new Int32Array(sab, 0, HEADER_BYTES / 4);
    Atomics.store(header, 0, BLOCK_LAYOUT);
    let offset = HEADER_BYTES;
    robots = instances.map((i) => {
        const bytes = slotBytes(i.geoms);
        const ring = { seq: [], t: [], poses: [] };
        for (let k = 0; k < RING_SLOTS; k++) {
            const at = offset + k * bytes;
            ring.seq.push(new Int32Array(sab, at, 1));
            ring.t.push(new Float64Array(sab, at + 8, 1));
            ring.poses.push(new Float32Array(sab, at + SLOT_POSES, i.geoms * POSE_STRIDE));
        }
        offset += RING_SLOTS * bytes;
        return { model: i.model, sha256: i.sha256.toLowerCase(), geoms: i.geoms, ring, next: 0, lastT: -Infinity };
    });
    if (config && config.stream) {
        stream(config.stream);
    } else {
        replay().catch((err) => fail(`replay: ${err.message || err}`));
    }
}

// Publish one sample for robot `r`, shown at absolute time `t`: into its next
// ring slot under that slot's seqlock (odd while writing, even when stable).
// `write(poses)` fills the slot's Float32Array. Single writer.
function publish(r, t, write) {
    const k = r.next;
    r.next = (k + 1) % RING_SLOTS;
    const seq = r.ring.seq[k];
    Atomics.store(seq, 0, (seq[0] + 1) | 1);
    r.ring.t[k][0] = t;
    write(r.ring.poses[k]);
    Atomics.store(seq, 0, seq[0] + 1);
    r.lastT = t;
    Atomics.add(header, 1, 1);
}

// ── replay ──────────────────────────────────────────────────────────────────

async function replay() {
    const clips = [];
    for (const r of robots) {
        const url = `${config.captures_base}/${r.model}.capture.json`;
        progress(`fetching ${r.model} capture…`);
        // no-cache: revalidate, so a re-recorded capture is never shadowed by a
        // stale cached one (whose fingerprint check would then pass or fail
        // for the wrong file).
        const resp = await fetch(url, { cache: "no-cache" });
        if (!resp.ok) {
            fail(`${url}: HTTP ${resp.status} — ${r.model} stays in its imported pose`);
            continue;
        }
        const cap = await resp.json();
        const problem =
            cap.format !== "awsm-mujoco-capture" ? `not a capture (format ${cap.format})`
            : (cap.source && cap.source.sha256 || "").toLowerCase() !== r.sha256
                ? `fingerprint mismatch: the capture is of ${cap.source && cap.source.filename} ` +
                  `(${(cap.source && cap.source.sha256 || "").slice(0, 12)}), the scene's ${r.model} ` +
                  `is ${r.sha256.slice(0, 12)} — re-record from the same USD`
            : cap.geom_count !== r.geoms ? `geom count ${cap.geom_count} ≠ the scene's ${r.geoms}`
            : !cap.frames || cap.frames.length < 2 ? "fewer than two frames"
            : null;
        if (problem) {
            fail(`${r.model}: ${problem}`);
            continue;
        }
        const n = cap.frames.length;
        const stride = r.geoms * POSE_STRIDE;
        const data = new Float32Array(n * stride);
        cap.frames.forEach((f, i) => data.set(f.geom_poses, i * stride));
        const dt = cap.frames[1].time - cap.frames[0].time;
        // A recorded loop's last frame leads into its first, so the period is
        // one step longer than the last timestamp.
        clips.push({ r, data, n, stride, dt, period: n * dt });
    }
    if (clips.length === 0) return;
    Atomics.store(header, 2, SOURCE_REPLAY);
    setSource("poses: looping recorded captures — scripted joint motion by forward kinematics, not physics");
    // Frame g of the endless loop is capture frame g mod n, due at t0 + g·dt.
    const t0 = nowAbs();
    for (const c of clips) {
        c.dtMs = c.dt * 1000;
        c.g = 0;
        // Never so far ahead that the writer comes round to the pair around
        // "now": the ring holds RING_SLOTS frames, and 3 of them stay behind.
        c.leadMs = Math.min(REPLAY_LEAD_MS, (RING_SLOTS - 3) * c.dtMs);
    }
    const tick = () => {
        const now = nowAbs();
        for (const c of clips) {
            // After a stall (a throttled worker) skip to the frame before now
            // rather than publish the backlog.
            const behind = Math.floor((now - t0) / c.dtMs) - 1;
            if (behind - c.g > RING_SLOTS) c.g = behind;
            while (t0 + c.g * c.dtMs <= now + c.leadMs) {
                const i = c.g % c.n;
                publish(c.r, t0 + c.g * c.dtMs, (out) => out.set(c.data.subarray(i * c.stride, (i + 1) * c.stride)));
                c.g++;
            }
        }
    };
    tick();
    setInterval(tick, REPLAY_TICK_MS);
}

// ── live stream ─────────────────────────────────────────────────────────────

function stream(url) {
    const bySha = new Map(robots.map((r) => [r.sha256, r]));
    let frames = 0;
    let lastReport = performance.now();
    let retry = 500;

    const connect = () => {
        setSource(`poses: live stream ${url} — connecting…`);
        let ws;
        try {
            ws = new WebSocket(url);
        } catch (err) {
            fail(`stream ${url}: ${err.message || err}`);
            return;
        }
        ws.binaryType = "arraybuffer";
        ws.onopen = () => {
            retry = 500;
            Atomics.store(header, 2, SOURCE_STREAM);
            for (const r of robots) r.stream = null; // a new connection: map its clock afresh
            setSource(`poses: live stream ${url} — connected, waiting for frames`);
        };
        ws.onmessage = (ev) => {
            if (!(ev.data instanceof ArrayBuffer)) return; // text = the producer's hello; ignored
            const arrival = nowAbs();
            const buf = ev.data;
            const r = buf.byteLength >= 32 ? bySha.get(hex(new Uint8Array(buf, 0, 32))) : undefined;
            const poseBytes = r ? r.geoms * POSE_STRIDE * 4 : 0;
            let src;
            let step = null;
            let dt = null;
            if (r && buf.byteLength === 40 + poseBytes) {
                const view = new DataView(buf, 32, 8);
                step = view.getUint32(0, true);
                dt = view.getFloat32(4, true) * 1000;
                src = new Float32Array(buf, 40);
            } else if (r && buf.byteLength === 32 + poseBytes) {
                src = new Float32Array(buf, 32);
            } else {
                Atomics.add(header, 3, 1);
                return;
            }
            const t = streamTime(r, arrival, step, dt);
            publish(r, t, (out) => out.set(src));
            frames++;
            const now = performance.now();
            if (now - lastReport > 1000) {
                const fps = (frames * 1000) / (now - lastReport);
                setSource(
                    `poses: live stream ${url} — ${fps.toFixed(0)} robot frames/s · ` +
                    `${Atomics.load(header, 3)} dropped`,
                );
                frames = 0;
                lastReport = now;
            }
        };
        ws.onclose = () => {
            // The robots hold their last pose; reconnect with backoff.
            setSource(`poses: live stream ${url} — disconnected, retrying in ${(retry / 1000).toFixed(1)} s`);
            setTimeout(connect, retry);
            retry = Math.min(retry * 2, 8000);
        };
        ws.onerror = () => { /* onclose follows and handles it */ };
    };
    connect();
}

// When a stream frame is shown (absolute ms). The producer's model time
// (step · dt, or one estimated dt per frame for a stepless stream) is mapped
// onto the page clock by the smallest recent (arrival − model time): those
// least-delayed arrivals are the closest thing to the producer's clock. A
// playout delay of PLAYOUT_STEPS steps then keeps the renderer supplied
// through ordinary network jitter. A step that goes backwards (the producer
// restarted) or a new dt starts the mapping afresh.
function streamTime(r, arrival, step, dt) {
    const s = r.stream || (r.stream = { model: 0, lastStep: null, dt: null, offsets: [], arrivals: [] });
    if (step === null) {
        // Plain form: dt from the median of recent arrival intervals.
        s.arrivals.push(arrival);
        if (s.arrivals.length > DT_WINDOW + 1) s.arrivals.shift();
        const gaps = s.arrivals.slice(1).map((a, i) => a - s.arrivals[i]).sort((a, b) => a - b);
        dt = gaps.length ? gaps[gaps.length >> 1] : 1000 / 30;
        s.model += s.lastStep === null ? 0 : dt;
        s.lastStep = 0;
    } else {
        const restart = s.lastStep === null || step <= s.lastStep || s.dt === null || Math.abs(dt - s.dt) > s.dt * 0.01;
        if (restart) {
            s.offsets = [];
            s.base = step;
        }
        s.model = (step - s.base) * dt;
        s.lastStep = step;
    }
    s.dt = dt;
    s.offsets.push(arrival - s.model);
    if (s.offsets.length > OFFSET_WINDOW) s.offsets.shift();
    let offset = Infinity;
    for (const o of s.offsets) offset = Math.min(offset, o);
    // Never earlier than the previous sample: a newly smallest offset must not
    // send time backwards.
    return Math.max(s.model + offset + PLAYOUT_STEPS * dt, r.lastT + dt * 0.25);
}

function hex(bytes) {
    let s = "";
    for (const b of bytes) s += b.toString(16).padStart(2, "0");
    return s;
}
