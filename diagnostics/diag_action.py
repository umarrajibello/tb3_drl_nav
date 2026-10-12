"""Stage 4 — Action realization audit."""
import numpy as np
from env.tb3_env import TB3Env

env = TB3Env(env_type='mixed', render_mode=None)

for action in [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.0, 0.5], [-1.0, 0.0]]:
    env.reset(seed=42)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
    v = info.get('transition_linear_velocity', None)
    w = info.get('transition_angular_velocity', None)
    print(f"Action {action}: realized v={v:.3f} m/s, w={w:.3f} rad/s")

env.close()