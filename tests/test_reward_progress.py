from env.reward import RewardCalculator


def main():
    reward = RewardCalculator()

    print("=== REWARD PROGRESS COMPONENT TEST ===")

    test_cases = [
        ("small forward progress", 0.001),
        ("normal forward progress", 0.002),
        ("larger forward progress", 0.010),
        ("zero progress", 0.000),
        ("backward progress", -0.010),
    ]

    for name, delta_s in test_cases:
        components = reward.get_components(
            prev_wp_dist=1.0,
            curr_wp_dist=1.0,
            progress_delta=delta_s,
        )

        print(
            f"{name:25s} "
            f"delta_s={delta_s:+.3f} "
            f"r_progress={components['progress']:+.3f}"
        )


if __name__ == "__main__":
    main()
