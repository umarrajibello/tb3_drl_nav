import numpy as np

from env.tb3_env_hybrid import TB3HybridEnv


def main():
    env = TB3HybridEnv()

    try:
        obs, info = env.reset(seed=0)

        robot_pose = env.get_robot_pose()

        print("\n=== HYBRID PATH-PROGRESS DIAGNOSTIC ===")

        print(
            f"Initial robot pose: "
            f"{robot_pose}"
        )

        print(
            f"Initial waypoint:   "
            f"{env.current_waypoint}"
        )

        print(
            f"Initial path points: "
            f"{len(env.path) if env.path is not None else 0}"
        )

        print(
            f"Initial lookahead:  "
            f"{env.current_lookahead:.3f}"
        )

        print(
            f"Initial path s:     "
            f"{info.get('path_progress')}"
        )

        print("\nColumns:")

        print(
            "step | "
            "robot_x | robot_y | "
            "path_s | delta_s | "
            "replan | lookahead | "
            "wp_dist | reward"
        )

        print("-" * 110)

        for step in range(50):

            # Moderate forward motion,
            # zero commanded angular velocity.
            action = np.array(
                [0.5, 0.0],
                dtype=np.float32,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            robot_pose = (
                env.get_robot_pose()
            )

            waypoint = np.asarray(
                env.current_waypoint,
                dtype=np.float32,
            )

            wp_dist = float(
                np.linalg.norm(
                    robot_pose[:2]
                    - waypoint[:2]
                )
            )

            path_s = info.get(
                "path_progress"
            )

            delta_s = info.get(
                "path_progress_delta"
            )

            replanned = info.get(
                "path_replanned",
                False,
            )

            print(
                f"{step:4d} | "
                f"{robot_pose[0]:7.3f} | "
                f"{robot_pose[1]:7.3f} | "
                f"{path_s if path_s is not None else float('nan'):7.3f} | "
                f"{delta_s:+7.3f} | "
                f"{str(replanned):6s} | "
                f"{info['lookahead_distance']:9.3f} | "
                f"{wp_dist:7.3f} | "
                f"{reward:+8.3f}"
            )

            if terminated or truncated:

                print("\nEpisode ended.")

                print(
                    f"terminated = "
                    f"{terminated}"
                )

                print(
                    f"truncated  = "
                    f"{truncated}"
                )

                break

    finally:
        env.close()


if __name__ == "__main__":
    main()