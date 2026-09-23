from env.reward import RewardCalculator
import numpy as np


def main():
    reward = RewardCalculator()

    print("=== INTEGRATED REWARD TEST ===")

    lidar_scan = np.full(36, 0.10, dtype=np.float32)

    components = reward.get_components(
        prev_wp_dist=1.0,
        curr_wp_dist=1.0,
        lidar_scan=lidar_scan,
        progress_delta=0.01,
        reached_goal=False,
        collision=False,
    )

    print(f"progress : {components['progress']:+.6f}")
    print(f"clearance: {components['clearance']:+.6f}")
    print(f"time     : {components['time']:+.6f}")
    print(f"success  : {components['success']:+.6f}")
    print(f"collision: {components['collision']:+.6f}")
    print(f"total    : {components['total']:+.6f}")

    expected = (
        components["progress"]
        + components["clearance"]
        + components["time"]
        + components["success"]
        + components["collision"]
    )

    print(f"sum      : {expected:+.6f}")
    print(f"difference: {components['total'] - expected:+.6f}")


if __name__ == "__main__":
    main()
