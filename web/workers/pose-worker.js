// Pose worker — the template's pose SOURCE. Publishes every robot's geom world
// poses into the SharedArrayBuffer pose block (layout: src/protocol.rs).
//
// Two sources, indistinguishable to the rest of the app:
//
//   replay (default)    loops media/captures/<model>.capture.json — a scripted
//                       joint sweep recorded offline by forward kinematics
//                       over the USD joints (awsm-renderer-isaac-record).
//                       Joint animation, not physics.
//   stream (?stream=)   binary frames from a WebSocket: 32 raw bytes of the
//                       robot's SHA-256 fingerprint, then 7 × geom_count f32
//                       little-endian [px,py,pz,qw,qx,qy,qz]. Fed by an Isaac
//                       Lab harness or harness/replay_stream.py.
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

const POSE_HEADER = 4;
const POSE_STRIDE = 7;
const SOURCE_REPLAY = 1;
const SOURCE_STREAM = 2;

let config = null;
let header = null; // Int32Array [seq, frames, source, dropped]
let poses = null;  // Float32Array over the pose span
let robots = [];   // {model, sha256, geoms, offset}

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
    header = new Int32Array(sab);
    poses = new Float32Array(sab, POSE_HEADER * 4);
    let offset = 0;
    robots = instances.map((i) => {
        const r = { model: i.model, sha256: i.sha256.toLowerCase(), geoms: i.geoms, offset };
        offset += i.geoms * POSE_STRIDE;
        return r;
    });
    if (config && config.stream) {
        stream(config.stream);
    } else {
        replay().catch((err) => fail(`replay: ${err.message || err}`));
    }
}

// Write under the seqlock: odd while writing, even when stable. Single writer.
function publish(write) {
    Atomics.store(header, 0, header[0] + 1);
    write();
    Atomics.store(header, 1, header[1] + 1);
    Atomics.store(header, 0, header[0] + 1);
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
    const t0 = performance.now();
    const tick = () => {
        const now = (performance.now() - t0) / 1000;
        publish(() => {
            for (const c of clips) {
                const t = now % c.period;
                const i = Math.floor(t / c.dt) % c.n;
                const j = (i + 1) % c.n;
                const a = (t - i * c.dt) / c.dt;
                blend(c.data, i * c.stride, j * c.stride, a, c.r);
            }
        });
    };
    tick();
    setInterval(tick, 1000 / 60);
}

// Frame i → frame j at fraction a: positions lerp, rotations nlerp along the
// short arc (q and -q are the same rotation).
function blend(src, i, j, a, r) {
    const out = r.offset;
    for (let g = 0; g < r.geoms; g++) {
        const p = g * POSE_STRIDE;
        for (let k = 0; k < 3; k++) {
            poses[out + p + k] = src[i + p + k] + (src[j + p + k] - src[i + p + k]) * a;
        }
        let dot = 0;
        for (let k = 3; k < 7; k++) dot += src[i + p + k] * src[j + p + k];
        const s = dot < 0 ? -1 : 1;
        let len = 0;
        for (let k = 3; k < 7; k++) {
            const v = src[i + p + k] * (1 - a) + s * src[j + p + k] * a;
            poses[out + p + k] = v;
            len += v * v;
        }
        len = Math.sqrt(len) || 1;
        for (let k = 3; k < 7; k++) poses[out + p + k] /= len;
    }
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
            setSource(`poses: live stream ${url} — connected, waiting for frames`);
        };
        ws.onmessage = (ev) => {
            if (!(ev.data instanceof ArrayBuffer)) return; // text = the producer's hello; ignored
            const buf = ev.data;
            const r = buf.byteLength >= 32 ? bySha.get(hex(new Uint8Array(buf, 0, 32))) : undefined;
            if (!r || buf.byteLength !== 32 + r.geoms * POSE_STRIDE * 4) {
                Atomics.add(header, 3, 1);
                return;
            }
            const src = new Float32Array(buf, 32);
            publish(() => poses.set(src, r.offset));
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

function hex(bytes) {
    let s = "";
    for (const b of bytes) s += b.toString(16).padStart(2, "0");
    return s;
}
