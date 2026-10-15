"""Literature-inspired reward for TurtleBot3 navigation.

The reward uses four interpretable components:

1. progress toward the task-specific navigation reference;
2. continuous obstacle-clearance penalty;
3. small per-step efficiency cost;
4. sparse terminal reward/penalty for success or collision.

Progress is represented in two compatible ways:

For TB3Env:
    progress = previous goal distance - current goal distance

For TB3HybridEnv:
    progress = current RRT path coordinate
               - previous RRT path coordinate

The hybrid environment therefore uses global-path advancement for the
progress reward while the adaptive waypoint remains the local target
provided to the SAC controller.
"""

import numpy as np


class RewardCalculator:
    # Dense shaping
    PROGRESS_WEIGHT = 2.0

    # LiDAR is normalized to [0, 1]. With LIDAR_RANGE=3.5 m,
    # 0.15 corresponds to about 0.525 m physical clearance.
    SAFETY_DISTANCE = 0.15
    CLEARANCE_WEIGHT = 1.0

    # Discourages unnecessarily long/looping episodes.
    STEP_PENALTY = 0.01

    # Sparse task-level rewards.
    SUCCESS_REWARD = 10.0
    COLLISION_PENALTY = 10.0

    # Protects the progress term from abnormal jumps.
    MAX_PROGRESS = 0.20

    def __init__(
        self,
        progress_weight=PROGRESS_WEIGHT,
        safety_distance=SAFETY_DISTANCE,
        clearance_weight=CLEARANCE_WEIGHT,
        step_penalty=STEP_PENALTY,
        success_reward=SUCCESS_REWARD,
        collision_penalty=COLLISION_PENALTY,
        max_progress=MAX_PROGRESS,
    ):
        self.progress_weight = float(progress_weight)
        self.safety_distance = float(safety_distance)
        self.clearance_weight = float(clearance_weight)
        self.step_penalty = float(step_penalty)
        self.success_reward = float(success_reward)
        self.collision_penalty = float(collision_penalty)
        self.max_progress = float(max_progress)

        self.step_count = 0

    def reset_step_count(self):
        """Reset the internal reward step counter."""
        self.step_count = 0

    # ------------------------------------------------------------------
    # Main reward calculation
    # ------------------------------------------------------------------

    def compute(
        self,
        prev_wp_dist,
        curr_wp_dist,
        robot_pose,
        path=None,
        collision=False,
        reached_goal=False,
        action=None,
        lidar_scan=None,
        goal_xy=None,
        progress_delta=None,
    ):
        """Return (reward, done).

        Parameters
        ----------
        prev_wp_dist:
            Previous distance to the current local target.

            Used by the base E2E environment.

        curr_wp_dist:
            Current distance to the current local target.

            Used by the base E2E environment.

        progress_delta:
            Optional direct progress measurement.

            When provided, this value is used instead of
            ``prev_wp_dist - curr_wp_dist``.

            TB3HybridEnv uses this for RRT path-coordinate progress:

                progress_delta = s_t - s_(t-1)

            The existing distance arguments remain in the interface
            for compatibility with TB3Env and existing code.
        """

        self.step_count += 1

        # --------------------------------------------------------------
        # 1. Progress
        # --------------------------------------------------------------
        #
        # E2E:
        #     previous goal distance - current goal distance
        #
        # Hybrid:
        #     current RRT path coordinate - previous RRT path coordinate
        #
        # The direct progress_delta argument allows both environments
        # to use the same reward calculator without changing the
        # overall reward structure.
        # --------------------------------------------------------------

        if progress_delta is None:
            prev_dist = float(prev_wp_dist)
            curr_dist = float(curr_wp_dist)

            progress = prev_dist - curr_dist

        else:
            progress = float(progress_delta)

        progress = float(
            np.clip(
                progress,
                -self.max_progress,
                self.max_progress,
            )
        )

        r_progress = (
            self.progress_weight
            * progress
        )

        # --------------------------------------------------------------
        # 2. Continuous safety penalty
        # --------------------------------------------------------------

        r_clearance = 0.0

        if lidar_scan is not None:
            scan = np.asarray(
                lidar_scan,
                dtype=np.float32,
            ).reshape(-1)

            if scan.size:
                min_clearance = float(
                    np.nanmin(scan)
                )

                min_clearance = float(
                    np.clip(
                        min_clearance,
                        0.0,
                        1.0,
                    )
                )

                if (
                    min_clearance
                    < self.safety_distance
                ):
                    ratio = (
                        self.safety_distance
                        - min_clearance
                    ) / max(
                        self.safety_distance,
                        1e-8,
                    )

                    r_clearance = (
                        -self.clearance_weight
                        * ratio
                    )

        # --------------------------------------------------------------
        # 3. Step/time cost
        # --------------------------------------------------------------

        r_time = -self.step_penalty

        # --------------------------------------------------------------
        # 4. Terminal outcome
        # --------------------------------------------------------------

        # Success takes precedence if both flags are accidentally true.
        if reached_goal:
            r_terminal = self.success_reward
            done = True

        elif collision:
            r_terminal = -self.collision_penalty
            done = True

        else:
            r_terminal = 0.0
            done = False

        reward = (
            r_progress
            + r_clearance
            + r_time
            + r_terminal
        )

        return float(reward), bool(done)

    # ------------------------------------------------------------------
    # Reward component diagnostics
    # ------------------------------------------------------------------

    def get_components(
        self,
        prev_wp_dist,
        curr_wp_dist,
        lidar_scan=None,
        collision=False,
        reached_goal=False,
        progress_delta=None,
    ):
        """Return individual reward components for debugging/evaluation.

        If ``progress_delta`` is supplied, it is used directly.

        Otherwise progress is calculated from local-target distance,
        preserving the original E2E behavior.
        """

        if progress_delta is None:
            progress = (
                float(prev_wp_dist)
                - float(curr_wp_dist)
            )
        else:
            progress = float(
                progress_delta
            )

        progress = float(
            np.clip(
                progress,
                -self.max_progress,
                self.max_progress,
            )
        )

        r_progress = (
            self.progress_weight
            * progress
        )

        # --------------------------------------------------------------
        # Clearance
        # --------------------------------------------------------------

        r_clearance = 0.0
        min_clearance = 1.0

        if lidar_scan is not None:
            scan = np.asarray(
                lidar_scan,
                dtype=np.float32,
            ).reshape(-1)

            if scan.size:
                min_clearance = float(
                    np.clip(
                        np.nanmin(scan),
                        0.0,
                        1.0,
                    )
                )

                if (
                    min_clearance
                    < self.safety_distance
                ):
                    ratio = (
                        self.safety_distance
                        - min_clearance
                    ) / max(
                        self.safety_distance,
                        1e-8,
                    )

                    r_clearance = (
                        -self.clearance_weight
                        * ratio
                    )

        # --------------------------------------------------------------
        # Time
        # --------------------------------------------------------------

        r_time = -self.step_penalty

        # --------------------------------------------------------------
        # Terminal components
        # --------------------------------------------------------------

        r_success = (
            self.success_reward
            if reached_goal
            else 0.0
        )

        r_collision = (
            -self.collision_penalty
            if collision
            and not reached_goal
            else 0.0
        )

        total = (
            r_progress
            + r_clearance
            + r_time
            + r_success
            + r_collision
        )

        return {
            "progress": float(r_progress),
            "clearance": float(r_clearance),
            "time": float(r_time),
            "success": float(r_success),
            "collision": float(r_collision),
            "total": float(total),

            # Raw progress quantity.
            #
            # E2E:
            #     distance reduction
            #
            # Hybrid:
            #     RRT path-coordinate advancement
            "raw_progress": float(progress),

            "min_lidar": float(
                min_clearance
            ),
        }