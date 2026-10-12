"""Repeat motion-realization test with multiple runs."""
import numpy as np
from env.tb3_env import TB3Env

env = TB3Env(env_type='mixed', render_mode=None)

test_actions = [
    (1.0, 0.0, 0.22, 0.0, "full forward"),
    (0.5, 0.0, 0.11, 0.0, "half forward"),
    (0.0, 0.5, 0.0, 1.42, "half right turn"),
    (0.0, 1.0, 0.0, 2.84, "full right turn"),
    (1.0, 0.5, None, None, "forward + turn"),
]

for ax, aw, expected_v, expected_w, label in test_actions:
    realized_vs = []
    realized_ws = []
    for trial in range(20):
        env.reset(seed=1000 + trial)
        for _ in range(50):  # let transients settle
            env.step(np.array([ax, aw], dtype=np.float32))
        v = env.get_robot_velocity()[0]
        w = env.get_robot_velocity()[1]
        realized_vs.append(v)
        realized_ws.append(w)

    print(f"{label}: action=({ax}, {aw})")
    print(f"  Realized v: mean {np.mean(realized_vs):.3f}, "
          f"std {np.std(realized_vs):.3f}")
    print(f"  Realized w: mean {np.mean(realized_ws):.3f}, "
          f"std {np.std(realized_ws):.3f}")
    if expected_v is not None:
        print(f"  Expected v: {expected_v}, ratio: {np.mean(realized_vs)/expected_v:.3f}")
    if expected_w is not None:
        print(f"  Expected w: {expected_w}, ratio: {np.mean(realized_ws)/expected_w:.3f}")

env.close()