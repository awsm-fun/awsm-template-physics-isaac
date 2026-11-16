"""The player's WebSocket wire, server side (see packages/frontend/src/protocol.rs).

One binary message per robot per step:

    bytes 0..32   raw SHA-256 of the robot's root USD (the scene instance's
                  fingerprint, `source.sha256` in its .mujoco.json)
    bytes 32..36  u32 LE  step number (monotonic)       } the "stepped" form;
    bytes 36..40  f32 LE  seconds per step (dt)          } omitted in the plain form
    then          7 x geom_count little-endian f32 [px,py,pz,qw,qx,qy,qz]

Send the stepped form when you can: the player maps step x dt onto its own
clock and shows each frame at its proper time, so network jitter never reaches
the screen. Plain frames still work; the player then counts each one as a step
and estimates dt from how often they arrive.

`PoseStreamServer` runs a WebSocket server on a background thread, so a sim
loop only calls `publish(...)` and never blocks on the network. Every
connected player gets every frame; a slow player simply sees fewer of them
(the newest pose is all that matters).

Needs `websockets` (pip install websockets).
"""

import asyncio
import hashlib
import struct
import threading

from websockets.asyncio.server import serve


def sha256_file(path):
    """Hex SHA-256 of a file: what the exporter writes as `source.sha256`."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame(sha256_hex, payload, step=None, dt=None):
    """Prefix a 7 x geom_count f32 payload with its robot's fingerprint and,
    given `step` and `dt` (seconds per step), the step header."""
    sha = bytes.fromhex(sha256_hex)
    if len(sha) != 32:
        raise ValueError(f"not a SHA-256: {sha256_hex!r}")
    if step is None or dt is None:
        return sha + bytes(payload)
    return sha + struct.pack("<If", step & 0xFFFFFFFF, dt) + bytes(payload)


class PoseStreamServer:
    def __init__(self, host="127.0.0.1", port=8765):
        self.host, self.port = host, port
        self.clients = set()
        self.loop = None
        self._ready = threading.Event()

    def start(self):
        threading.Thread(target=self._run, name="pose-stream", daemon=True).start()
        self._ready.wait()
        print(f"pose stream on ws://{self.host}:{self.port}")
        return self

    def _run(self):
        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async def handler(ws):
            self.clients.add(ws)
            print(f"player connected ({len(self.clients)})")
            try:
                await ws.send("awsm-pose-stream 1")  # text hello; the player ignores it
                await ws.wait_closed()
            finally:
                self.clients.discard(ws)
                print(f"player left ({len(self.clients)})")

        async with serve(handler, self.host, self.port, max_size=None, compression=None):
            self._ready.set()
            await asyncio.Future()

    def publish(self, frames):
        """Send a list of `frame(...)` bytes (one per robot) to every player.
        Thread-safe; returns immediately."""
        frames = list(frames)

        def send():
            for ws in list(self.clients):
                # Drop instead of queueing behind a slow player.
                if ws.transport.get_write_buffer_size() > (1 << 20):
                    continue
                for f in frames:
                    asyncio.ensure_future(ws.send(f))

        self.loop.call_soon_threadsafe(send)
