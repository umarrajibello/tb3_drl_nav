"""Stage 2 — Reward component audit over a random episode."""
import numpy as np
from env.tb3_env_hybrid import TB3HybridEnv

env = TB3HybridEnv(env_type='mixed', render_mode=None, lookahead_mode='fixed')

for ep in range(3):
    obs, info = env.reset(seed=42 + ep)
    done = False
    steps = 0
    totals = {'progress': 0.0, 'clearance': 0.0, 'time': 0.0, 
              'success': 0.0, 'collision': 0.0}
    while not done and steps < 5000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        comps = info.get('reward_components', {})
        for k in totals:
            totals[k] += comps.get(k, 0.0)
        steps += 1
    print(f"Episode {ep}: {steps} steps")
    print(f"  total reward: {sum(totals.values()):.2f}")
    for k, v in totals.items():
        print(f"  {k}: {v:.2f}")

env.close()