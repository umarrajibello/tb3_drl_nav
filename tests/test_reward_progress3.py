from env.reward import RewardCalculator
import numpy as np


def main():
    reward = RewardCalculator()

    print("=== CLEARANCE PENALTY TEST ===")

    test_cases = [
        ("safe", 1.00),
        ("near threshold", 0.20),
        ("at threshold", 0.15),
        ("close obstacle", 0.10),
        ("very close", 0.05),
        ("zero clearance", 0.00),
    ]

    for name, lidar_value in test_cases:
        lidar_scan = np.full(36, lidar_value, dtype=np.float32)

        components = reward.get_components(
            prev_wp_dist=1.0,
            curr_wp_dist=1.0,
            lidar_scan=lidar_scan,
            progress_delta=0.0,
        )

        print(
            f"{name:20s} "
            f"min_lidar={components['min_lidar']:.3f} "
            f"clearance={components['clearance']:+.3f} "
            f"total={components['total']:+.3f}"
        )


if __name__ == "__main__":
    main()
