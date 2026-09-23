from env.tb3_env_hybrid import TB3HybridEnv


def main():
    print("=== HYBRID RRT REPLANNING REWARD TEST ===")

    env = TB3HybridEnv(render_mode=None)

    obs, info = env.reset()

    print()
    print(
        "step | path_s | delta_s | progress_reward | "
        "replan | error"
    )
    print("-" * 75)

    replan_count = 0

    for step in range(35):

        action = [0.2, 0.0]

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        path_s = info["path_progress"]
        delta_s = info["path_progress_delta"]
        replanned = info["path_replanned"]

        expected_progress = (
            2.0
            * max(
                min(delta_s, 0.20),
                -0.20,
            )
        )

        components = env.reward_calc.get_components(
            prev_wp_dist=0.0,
            curr_wp_dist=0.0,
            progress_delta=delta_s,
        )

        actual_progress = components["progress"]

        error = (
            actual_progress
            - expected_progress
        )

        print(
            f"{step:4d} | "
            f"{path_s if path_s is not None else 0.0:+.6f} | "
            f"{delta_s:+.6f} | "
            f"{actual_progress:+.6f} | "
            f"{str(replanned):5s} | "
            f"{error:+.6f}"
        )

        if replanned:
            replan_count += 1

            # Critical assertion:
            # a newly accepted RRT path must not inherit
            # progress from the previous path.
            assert abs(delta_s) < 1e-8, (
                f"Replan at step {step} produced "
                f"non-zero delta_s={delta_s}"
            )

            assert abs(actual_progress) < 1e-8, (
                f"Replan at step {step} produced "
                f"non-zero progress reward="
                f"{actual_progress}"
            )

        if terminated or truncated:
            break

    env.close()

    print()
    print(
        f"RRT replans observed: {replan_count}"
    )

    assert replan_count > 0, (
        "No RRT replanning event was observed."
    )

    print(
        "RESULT: PASS — every observed replan "
        "had zero progress reward."
    )


if __name__ == "__main__":
    main()
