
import pybullet as p
import time
from env.tb3_env import TB3Env

# Force GUI mode BEFORE importing anything else that uses config
import config
config.GUI = True

# Now create environment
#env = TB3Env()
#env.reset()

from env.tb3_env_hybrid1 import TB3HybridEnv
env = TB3HybridEnv(env_type="dense", gui=True)
obs = env.reset()
print("Reset done")
action = [0.0, 0.0]   # zero action
obs, reward, done, info = env.step(action)
print(f"After step: done={done}, reward={reward}, info={info}")



print("Robot moving forward for 500 steps...")
for i in range(500):
    # Linear = 0.1 m/s, Angular = 0
    action = [0.1/0.22, 0.0]   # scale to action range [0,1] and [-1,1]
    obs, reward, done, _ = env.step(action)
    pos = env.get_robot_position()
    print(f"Step {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}), reward={reward:.2f}")
    if done:
        break
    time.sleep(0.02)   # slow down for visualization

input("Press Enter to close...")
p.disconnect()