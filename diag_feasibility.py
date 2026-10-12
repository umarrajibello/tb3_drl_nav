"""Stage 1 — Random-policy feasibility check."""
import numpy as np
from env.tb3_env import TB3Env

env = TB3Env(env_type='mixed', render_mode=None)

successes = 0
timeouts = 0
collisions = 0
goal_distances = []
episodes = 100

for ep in range(episodes):
    obs, info = env.reset(seed=42 + ep)
    start_dist = info.get('distance_to_goal', None)
    goal_distances.append(start_dist)
    done = False
    steps = 0
    while not done and steps < 5000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    if info.get('reached_goal', False):
        successes += 1
    elif info.get('collision', False):
        collisions += 1
    else:
        timeouts += 1

env.close()

print(f"Random-policy over {episodes} episodes:")
print(f"  Successes: {successes}")
print(f"  Collisions: {collisions}")
print(f"  Timeouts: {timeouts}")
print(f"  Goal distance min: {min(goal_distances):.2f} m")
print(f"  Goal distance max: {max(goal_distances):.2f} m")
print(f"  Goal distance mean: {np.mean(goal_distances):.2f} m")