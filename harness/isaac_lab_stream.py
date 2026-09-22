"""Stream an Isaac Lab robot to the physics-isaac player (reference code).

Drop this directory next to an Isaac Lab script and call `IsaacLabStream.step()`
after each `env.step(...)`:

    from isaac_lab_stream import IsaacLabStream

    stream = IsaacLabStream(
        robot=env.unwrapped.scene["robot"],          # isaaclab.assets.Articulation
        env_origins=env.unwrapped.scene.env_origins,
        sidecar="panda.mujoco.json",                  # from awsm-renderer-isaac-export
        usd="/path/to/the/franka.usd",                # the SAME root USD the sim loaded
    ).start()                                         # ws://0.0.0.0:8765
    while simulation_app.is_running():
        obs, *_ = env.step(policy(obs))
        stream.step()

Then open the player with `?stream=ws://<sim host>:8765`.

The fingerprint is the SHA-256 of the root USD file, the same bytes the
exporter hashed. If the sim loads a different file than the one the player's
scene was exported from, the player drops every frame (and says so) rather
than drive the wrong robot. Frames are env-local: env `env_id`'s origin is
subtracted, so the robot stands where it was authored.

Needs numpy and websockets. NOT yet run inside Isaac Lab (that needs a
Linux/Windows machine with an RTX GPU); the pieces it is built from are
verified: the frame builder offline against the exporter's fixtures, and the
wire end to end by `replay_stream.py`.
"""

from isaac_geom_frame import GeomFrame
from pose_stream import PoseStreamServer, frame, sha256_file


class IsaacLabStream:
    def __init__(self, robot, env_origins, sidecar, usd, env_id=0, host="0.0.0.0", port=8765, every=1):
        self.robot = robot
        self.env_origins = env_origins
        self.env_id = env_id
        self.every = max(1, every)  # stream every Nth sim step
        self.sha = sha256_file(usd)
        self.geom_frame = GeomFrame(sidecar, robot.data.body_names)
        self.server = PoseStreamServer(host, port)
        self.steps = 0
        print(f"streaming {usd} ({self.sha[:12]}), env {env_id}, bodies {len(robot.data.body_names)}")

    def start(self):
        self.server.start()
        return self

    def step(self):
        self.steps += 1
        if self.steps % self.every or not self.server.clients:
            return
        pose = self.robot.data.body_link_pose_w[self.env_id].cpu().numpy().copy()  # (B, 7) world, wxyz
        pose[:, :3] -= self.env_origins[self.env_id].cpu().numpy()                 # env-local
        self.server.publish([frame(self.sha, self.geom_frame(pose))])
