from env.tb3_env_hybrid import TB3HybridEnv


def main():
    print("=== HYBRID PROGRESS -> REWARD LINKAGE TEST ===")

    env = TB3HybridEnv(render_mode=None)

    obs, info = env.reset()

    print()
    print(
        "step | delta_s | expected_rp | actual_rp | "
        "error | replan"
    )
    print("-" * 72)

    for step in range(15):

        # Small forward command.
        # This is only for generating actual environment motion.
        action = [0.2, 0.0]

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        delta_s = float(
            info["path_progress_delta"]
        )

        # Expected progress reward from the current reward definition.
        expected_progress = (
            2.0
            * max(
                min(delta_s, 0.20),
                -0.20,
            )
        )

        # Reconstruct the actual reward components using the
        # same RewardCalculator instance and the progress_delta
        # reported by the environment.
        components = env.reward_calc.get_components(
            prev_wp_dist=0.0,
            curr_wp_dist=0.0,
            lidar_scan=env.get_lidar(),
            collision=info["collision"],
            reached_goal=info["reached_goal"],
            progress_delta=delta_s,
        )

        actual_progress = float(
            components["progress"]
        )

        error = (
            actual_progress
            - expected_progress
        )

        print(
            f"{step:4d} | "
            f"{delta_s:+.6f} | "
            f"{expected_progress:+.6f} | "
            f"{actual_progress:+.6f} | "
            f"{error:+.6f} | "
            f"{str(info['path_replanned']):5s}"
        )

        if terminated or truncated:
            print()
            print("Episode terminated early.")
            break

    env.close()


if __name__ == "__main__":
    main()
