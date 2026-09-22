# AWSM Template
## Isaac Sim / Isaac Lab robots — w/ multithreaded rendering

_Deploys to GitHub Pages on every merge to `main`._

----
A copyable **template** from the [**Awsm**](https://awsm.fun) project: it
*plays* NVIDIA **Isaac Sim / Isaac Lab** robots, exported from their USD
assets, in [`awsm-renderer`](https://crates.io/crates/awsm-renderer), and
drives them from the outside through shared memory. Two robots ship: the
Franka **Panda** arm and the ANYbotics **ANYmal-D** quadruped, both from the
Isaac 5.0 asset library, with their Omniverse materials and textures.

Isaac Sim cannot run in a browser (it needs Linux or Windows and an RTX GPU).
So this template is the *other* end of an Isaac integration. The robots are
exported offline, authored into a scene in the [scene editor](https://scene.awsm.fun),
and moved at runtime by body poses that arrive from somewhere else:

- **Replay** (the default) loops a recorded capture per robot. The shipped
  captures are a scripted joint sweep, computed by forward kinematics over the
  robots' USD joints. It is joint animation, not physics: nothing collides and
  the quadruped does not walk.
- **Stream** (`?stream=ws://host:port`) takes live frames over a WebSocket,
  from an Isaac Lab process (`harness/isaac_lab_stream.py`) or from the mock
  that replays the captures over a socket (`task stream`).

The renderer never learns which one it is. It is the same "pose sink" as the
[`physics-mujoco`](https://github.com/awsm-fun/awsm-template-physics-mujoco)
template, minus the simulator in the page.

## Run it

```sh
task dev   # trunk serve on http://127.0.0.1:9000 (COOP/COEP enabled)
           # + a side media server on :9001
```

Besides the usual Rust-wasm toolchain (`task`, `trunk`, nightly via
`rust-toolchain.toml`), building needs a **wasm-capable clang** for the
renderer's meshopt C++ decode path. Apple's clang has no wasm backend, so on
macOS `brew install llvm` (the Taskfile finds the Homebrew path). Linux distro
clang works as-is.

Open it in a browser with **WebGPU** and **`SharedArrayBuffer`** (recent
Chrome/Edge). Drag to orbit, wheel to zoom.

### Live stream, locally

```sh
pip install websockets
task stream                       # ws://127.0.0.1:8765, both captures at 30 Hz
task stream -- --bogus            # also sends frames for an unknown robot
```

Then open `http://127.0.0.1:9000/?stream=ws://127.0.0.1:8765`. The HUD shows
the frame rate and how many frames were dropped. Stop the producer and the
robots hold their last pose while the page reconnects.

## How it's wired

| thread | code | role |
|---|---|---|
| **Main** | `src/main_thread.rs` | DOM (Dominator), spawns both workers, brokers startup. |
| **Render** | `src/render_thread.rs` | `awsm-renderer` on an `OffscreenCanvas`. Loads the player bundle, then each frame applies every robot's poses with `scene_loader::mujoco::apply_geom_poses`. |
| **Pose worker** | `web/workers/pose-worker.js` | The pose *source*: replays captures or reads the socket, and publishes into a `SharedArrayBuffer` pose block. |

Main and render share one wasm module and one `WebAssembly.Memory` (the
threaded build: nightly, `+atomics`, `build-std`, COOP/COEP; see
`Taskfile.yml`). The pose block is guarded by a **seqlock**: the writer makes
a counter odd, writes, makes it even, and the reader retries a torn frame
instead of blocking. The layout and the wire are documented in
`src/protocol.rs`.

### The one rule: the fingerprint

Every robot in the scene carries the SHA-256 of the root USD file it was
exported from. Every capture records it, and every stream frame starts with
it:

```text
bytes 0..32   SHA-256 of the robot's root USD, raw
bytes 32..    7 × geom_count little-endian f32: [px,py,pz,qw,qx,qy,qz] per geom
```

A capture of a different file is refused with an error on the HUD. A stream
frame for a robot the scene does not contain, or of the wrong length, is
dropped and counted. Nothing is ever applied to "whichever robot is first".
Several robots share one socket.

## Bring your own robot

Everything happens in the [`awsm-renderer`](https://github.com/awsm-fun/awsm-renderer)
repo's tools and the editor; this template only plays the result. See its
[`docs/isaac.md`](https://github.com/awsm-fun/awsm-renderer/blob/main/docs/isaac.md)
for the details.

1. **Export** the USD to a sidecar and a geometry GLB:
   ```sh
   cargo run --release -p awsm-renderer-isaac-export-cli -- robot.usd -o out/
   ```
2. **Import** `out/<name>.mujoco.json` in the scene editor (MuJoCo → import
   from URL), place it, and export a **player bundle** into `media/bundle/`.
3. **Record** a capture, or stream from Isaac Lab:
   ```sh
   cargo run --release -p awsm-renderer-isaac-export-cli \
     --bin awsm-renderer-isaac-record -- robot.usd -o media/captures/
   ```
   The capture must be named after the model (`<name>.capture.json`), which
   is the default. Use the same `--variant` flags as the export, or the
   fingerprint and geom count will not match.

The shipped captures were recorded with `--amplitude 0.8` (Panda) and the
default `0.4` (ANYmal).

## Streaming from Isaac Lab

`harness/` is the sim side, in plain Python with numpy and `websockets`:

| file | what |
|---|---|
| `isaac_geom_frame.py` | Isaac Lab body poses to the geom frame the renderer takes (a copy of the renderer repo's reference). |
| `pose_stream.py` | The WebSocket server, on a background thread, that never blocks the sim loop. |
| `isaac_lab_stream.py` | The hook: construct it with the articulation, the sidecar and the root USD, then call `step()` after each `env.step`. |
| `replay_stream.py` | The mock producer behind `task stream`. |

```python
from isaac_lab_stream import IsaacLabStream

stream = IsaacLabStream(
    robot=env.unwrapped.scene["robot"],
    env_origins=env.unwrapped.scene.env_origins,
    sidecar="panda.mujoco.json",
    usd="/path/to/franka.usd",     # the same file the scene was exported from
).start()                          # ws://0.0.0.0:8765
while simulation_app.is_running():
    obs, *_ = env.step(policy(obs))
    stream.step()
```

What is verified: the frame builder against the exporter's fixtures, the
hook with a stand-in articulation (rest pose reproduced exactly, fingerprint
matching), and the whole wire from Python to the page with `task stream`.
What is **not** verified: running the hook inside Isaac Lab, which needs a
Linux or Windows machine with an RTX GPU.

## Deploy

`task build` writes a static site to `dist/` with relative URLs, so it works
at a domain root, a GitHub Pages project path, or a CDN subpath. Static hosts
cannot send COOP/COEP headers, so `coi-serviceworker.js` re-imposes them on
first load.

## Credits

- Franka Panda and ANYmal-D USD assets: NVIDIA Isaac Sim 5.0 asset library
  (the robots' own licenses apply; see NVIDIA's Isaac Sim asset terms).
- Basis Universal transcoder (Apache-2.0), vendored at `web/vendor/basis`.

## License

MIT OR Apache-2.0, for the template's own code.
