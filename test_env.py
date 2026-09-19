import gym
from env.tb3_env import TB3Env

env = TB3Env(env_type='static', gui=False)
obs = env.reset()
for i in range(100):
    action = [0.0, 0.0]   # zero action
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f"Step {i}, reward {reward}")
    if terminated or truncated:
        break
print("Environment step test completed")

from stable_baselines3 import SAC
import numpy as np

model = SAC.load("./models/static_sac_end2end_seed42_balanced_reward_test_final/model.zip")
obs = np.random.randn(41).astype(np.float32)   # dummy observation
action, _ = model.predict(obs, deterministic=True)
print("Action:", action)
