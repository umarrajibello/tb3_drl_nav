import numpy as np

from env.tb3_env_hybrid import TB3HybridEnv


def main():
    env = TB3HybridEnv()

    try:
        obs, info = env.reset(seed=0)

        robot_pose = env.get_robot_pose()

        print("\n=== HYBRID PROGRESS DIAGNOSTIC ===")
        print(f"Initial robot pose: {robot_pose}")

        # Current waypoint is part of the hybrid navigation state.
        waypoint = np.asarray(env.current_waypoint, dtype=np.float32)

        print(f"Initial waypoint:   {waypoint}")

        # Number of points in the current RRT path.
        print(f"Initial path points: {len(env.path)}")

        print(f"Initial lookahead:  {env.current_lookahead:.3f}")

        # ---------------------------------------------------------
        # Initial projection of robot onto RRT path.
        #
        # _project_robot_onto_path() returns:
        #
        #   segment_index
        #   interpolation
        #   projection_point
        #   path_arc_length
        # ---------------------------------------------------------
        #_, _, _, previous_arc = env._project_robot_onto_path(
        #    np.asarray(robot_pose[:2], dtype=np.float32)
        #)
        previous_arc = env.debug_robot_arc



        print(f"Initial path coordinate s: {previous_arc:.3f}")

        print("\nColumns:")
        print(
            "step | "
            "robot_x | robot_y | "
            "s | delta_s | "
            "lookahead | "
            "wp_dist | "
            "reward"
        )

        print("-" * 100)

        for step in range(50):

            # -----------------------------------------------------
            # Manual control:
            #
            # action[0] = normalized linear velocity
            # action[1] = normalized angular velocity
            #
            # Moderate forward motion, zero commanded rotation.
            # -----------------------------------------------------
            action = np.array(
                [0.5, 0.0],
                dtype=np.float32,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            # -----------------------------------------------------
            # Robot pose
            # -----------------------------------------------------
            robot_pose = env.get_robot_pose()

            robot_x = float(robot_pose[0])
            robot_y = float(robot_pose[1])

            # -----------------------------------------------------
            # Current waypoint
            # -----------------------------------------------------
            waypoint = np.asarray(
                env.current_waypoint,
                dtype=np.float32,
            )

            waypoint_x = float(waypoint[0])
            waypoint_y = float(waypoint[1])

            # -----------------------------------------------------
            # Robot-to-waypoint distance
            # -----------------------------------------------------
            wp_dist = float(
                np.linalg.norm(
                    np.array(
                        [robot_x, robot_y],
                        dtype=np.float32,
                    )
                    - waypoint[:2]
                )
            )

            # -----------------------------------------------------
            # Project robot onto RRT path
            # -----------------------------------------------------
            #_, _, _, current_arc = env._project_robot_onto_path(
             #   np.asarray(
              #      [robot_x, robot_y],
               #     dtype=np.float32,
                #)
            #)
            current_arc = env.debug_robot_arc

            delta_arc = current_arc - previous_arc

            
            previous_arc = current_arc

            # -----------------------------------------------------
            # Print
            # -----------------------------------------------------
            print(
                f"{step:4d} | "
                f"{robot_x:7.3f} | "
                f"{robot_y:7.3f} | "
                f"{current_arc:6.3f} | "
                f"{delta_arc:+7.3f} | "
                f"{env.current_lookahead:9.3f} | "
                f"{wp_dist:7.3f} | "
                f"{reward:+8.3f}"
            )

            if terminated or truncated:
                print("\nEpisode ended.")
                print(f"terminated = {terminated}")
                print(f"truncated  = {truncated}")
                break

    finally:
        env.close()


if __name__ == "__main__":
    main()
