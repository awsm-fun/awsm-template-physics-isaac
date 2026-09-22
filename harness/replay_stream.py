"""Mock pose producer: streams recorded captures over the player's WebSocket
wire, so the stream path can be exercised without Isaac Sim.

    pip install websockets
    python harness/replay_stream.py media/captures/*.capture.json
    # then open the player with ?stream=ws://127.0.0.1:8765

Each capture is sent with the fingerprint it records (`source.sha256`), so the
player binds it to the matching robot exactly as it would a live Isaac Lab
frame. `--bogus` also sends one frame per second with an unknown fingerprint,
which the player must drop and count (the HUD shows "N dropped").
"""

import argparse
import json
import struct
import time

from pose_stream import PoseStreamServer, frame


def load(path):
    cap = json.load(open(path))
    if cap.get("format") != "awsm-mujoco-capture":
        raise SystemExit(f"{path}: not a capture")
    n = cap["geom_count"] * 7
    frames = []
    for f in cap["frames"]:
        if len(f["geom_poses"]) != n:
            raise SystemExit(f"{path}: frame at t={f['time']} has {len(f['geom_poses'])} floats, want {n}")
        frames.append(frame(cap["source"]["sha256"], struct.pack(f"<{n}f", *f["geom_poses"])))
    dt = cap["frames"][1]["time"] - cap["frames"][0]["time"]
    print(f"{path}: {cap['source']['filename']} {cap['source']['sha256'][:12]} "
          f"{cap['geom_count']} geoms, {len(frames)} frames @ {1 / dt:.0f} Hz")
    return frames, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bogus", action="store_true", help="also send an unknown-fingerprint frame every second")
    args = ap.parse_args()

    clips = [load(p) for p in args.captures]
    dt = min(d for _, d in clips)
    server = PoseStreamServer(args.host, args.port).start()
    t0 = time.monotonic()
    step = 0
    while True:
        t = step * dt
        batch = [frames[int(t / d) % len(frames)] for frames, d in clips]
        if args.bogus and step % max(1, round(1 / dt)) == 0:
            batch.append(b"\xff" * 32 + batch[0][32:])
        server.publish(batch)
        step += 1
        time.sleep(max(0.0, t0 + step * dt - time.monotonic()))


if __name__ == "__main__":
    main()
