from env.reward import RewardCalculator


def main():
    reward = RewardCalculator()

    print("=== REWARD PROGRESS CLIPPING TEST ===")

    test_cases = [
        ("exact positive limit", 0.20),
        ("above positive limit", 0.50),
        ("very large positive", 1.00),
        ("exact negative limit", -0.20),
        ("below negative limit", -0.50),
        ("very large negative", -1.00),
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
