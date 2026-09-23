from env.reward import RewardCalculator


def main():
    reward = RewardCalculator()

    print("=== TERMINAL REWARD TEST ===")

    test_cases = [
        ("normal", False, False),
        ("success", True, False),
        ("collision", False, True),
    ]

    for name, reached_goal, collision in test_cases:
        components = reward.get_components(
            prev_wp_dist=1.0,
            curr_wp_dist=1.0,
            progress_delta=0.0,
            reached_goal=reached_goal,
            collision=collision,
        )

        print(
            f"{name:12s} "
            f"progress={components['progress']:+.3f} "
            f"clearance={components['clearance']:+.3f} "
            f"time={components['time']:+.3f} "
            f"success={components['success']:+.3f} "
            f"collision={components['collision']:+.3f} "
            f"total={components['total']:+.3f}"
        )


if __name__ == "__main__":
    main()
