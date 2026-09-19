from env.tb3_env import TB3Env
import numpy as np
import time

env = TB3Env(env_type="empty", gui=True)

obs, _ = env.reset()

for step in range(1000):

    pose = env.get_robot_pose()
    goal = env.goal_pos

    dx = goal[0] - pose[0]
    dy = goal[1] - pose[1]

    target_yaw = np.arctan2(dy, dx)
    yaw_error = target_yaw - pose[2]
    yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

    linear = 0.5
    angular = 2.0 * yaw_error

    action = np.array([
        linear,
        np.clip(angular, -1, 1)
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    dist = np.hypot(dx, dy)

    print(
        f"step={step} "
        f"dist={dist:.2f} "
        f"yaw_err={yaw_error:.2f} "
        f"reward={reward:.2f}"
    )

    if terminated or truncated:
        print("DONE:", info)
        break

    time.sleep(0.01)
