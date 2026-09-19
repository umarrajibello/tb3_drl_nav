import gym
from env.tb3_env_hybrid import TB3HybridEnv

env = TB3HybridEnv(env_type='static', gui=False)
obs = env.reset()
for i in range(100):
    action = [0.0, 0.0]   # zero action
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f"Step {i}, reward {reward}")
    if terminated or truncated:
        break
print("Environment step test completed")
