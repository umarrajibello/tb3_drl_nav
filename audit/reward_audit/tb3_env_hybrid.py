"""Hybrid RRT-guided local navigation environment.

Committed thesis architecture:

    Static map
        ↓
    RRT global path
        ↓
    Adaptive path-distance lookahead
        ↓
    Local waypoint
        ↓
    LiDAR history + robot state + waypoint
        ↓
    Temporal encoder
    (CNN + LSTM/GRU/Transformer)
        ↓
    SAC
        ↓
    linear/angular velocity

RRT provides soft global guidance only.
SAC remains the local controller.

Dynamic obstacles are handled locally through LiDAR.

Reward design:
    - progress is measured using advancement along the current
      RRT path;
    - the adaptive waypoint remains an observation target;
    - progress reference is re-anchored whenever a new RRT path
      is accepted.
"""

import time

import numpy as np
import pybullet as p

from config import (
    LINEAR_VEL,
    ANGULAR_VEL,
    MAX_STEPS,
    PLANNER_UPDATE_FREQ,
    STATIC_OBSTACLES,
    WAYPOINT_DISTANCE,
)

from env.planners import RRTPlanner
from env.reward import RewardCalculator
from env.tb3_env import TB3Env


class TB3HybridEnv(TB3Env):
    """RRT-guided local navigation environment."""

    def __init__(
        self,
        render_mode=None,
        env_type=None,
        lookahead_mode="velocity_clearance",

    ):
        super().__init__(
            render_mode=render_mode,
            env_type=env_type,

        )

        # --------------------------------------------------------------
    # Adaptive lookahead configuration
    # --------------------------------------------------------------

        valid_modes = {
            "fixed",
            "velocity",
            "velocity_clearance",
        }

        if lookahead_mode not in valid_modes:
            raise ValueError(
                f"Invalid lookahead_mode={lookahead_mode!r}. "
                f"Expected one of {sorted(valid_modes)}."
            )

        self.lookahead_mode = lookahead_mode

        # --------------------------------------------------------------
        # Global planner
        # --------------------------------------------------------------

        self.planner = RRTPlanner(
            static_obstacles=STATIC_OBSTACLES,
            bounds=self.world_bounds,
        )

        # Current global path.
        self.path = None
        self.path_index = 0

        # --------------------------------------------------------------
        # Local waypoint
        # --------------------------------------------------------------

        self.current_waypoint = None
        self.current_lookahead = WAYPOINT_DISTANCE

        # Waypoint completion threshold.
        self.wp_threshold = 0.20

        # --------------------------------------------------------------
        # Planner timing
        # --------------------------------------------------------------

        self.planner_counter = 0

        # --------------------------------------------------------------
        # Reward progress reference
        # --------------------------------------------------------------
        #
        # E2E uses distance-to-goal.
        #
        # Hybrid uses RRT path coordinate:
        #
        #     s_t = arc-length position of robot on current RRT path
        #
        # prev_path_progress stores s_(t-1).
        #
        # It is reset/re-anchored whenever a new path is accepted.
        # --------------------------------------------------------------

        self.prev_path_progress = None

        # Retained for compatibility/debugging with older code.
        self.prev_dist_to_wp = None

        # Same reward implementation as the base environment.
        self.reward_calc = RewardCalculator()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """Reset base environment, plan RRT path, and initialize waypoint."""

        _, info = super().reset(
            seed=seed,
            options=options,
        )

        self.path = None
        self.path_index = 0
        self.current_waypoint = None
        self.current_lookahead = WAYPOINT_DISTANCE
        self.planner_counter = 0

        self.prev_path_progress = None
        self.prev_dist_to_wp = None

        # --------------------------------------------------------------
        # Initial global planning
        # --------------------------------------------------------------

        path_available = self._plan_global_path()

        robot_pose = self._get_robot_pose()
        v, _ = self.get_robot_velocity()
        lidar = self.get_lidar()

        # Select initial local waypoint.
        self._update_waypoint(
            robot_pose,
            v,
            lidar,
        )

        # --------------------------------------------------------------
        # Initialize path-progress reference.
        #
        # The initial state does not constitute progress.
        # Therefore we store the current path coordinate as the
        # reference for the first action.
        # --------------------------------------------------------------

        current_path_progress = (
            self._get_current_path_progress(
                robot_pose
            )
        )

        if current_path_progress is not None:
            self.prev_path_progress = (
                current_path_progress
            )

        else:
            self.prev_path_progress = None

        # Retain the local-waypoint distance for compatibility/debugging.
        self.prev_dist_to_wp = (
            self._waypoint_distance(
                robot_pose
            )
        )

        obs = self._get_hybrid_observation()

        info = dict(info)

        info.update(
            {
                "planner": "RRT",

                "path_available": (
                    self.path is not None
                ),

                "path_planning_success": bool(
                    path_available
                ),

                "waypoint_source": (
                    "rrt_path"
                    if self.path is not None
                    else "goal_fallback"
                ),
                
                "lookahead_mode": self.lookahead_mode,

                "lookahead_distance": float(
                    self.current_lookahead
                ),

                "path_progress": (
                    float(current_path_progress)
                    if current_path_progress is not None
                    else None
                ),

                "progress_mode": (
                    "rrt_path_arc"
                    if current_path_progress is not None
                    else "goal_fallback"
                ),
            }
        )

        return obs, info

    def step(self, action):
        """Execute one SAC local-control action."""

        self.step_count += 1
        self.planner_counter += 1

        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        )

        linear = (
            float(action[0])
            * LINEAR_VEL
        )

        angular = (
            float(action[1])
            * ANGULAR_VEL
        )

        # --------------------------------------------------------------
        # Local SAC control.
        #
        # RRT does not select the action.
        # SAC remains the local controller.
        # --------------------------------------------------------------

        left_speed = (
            linear
            - angular * self.WHEEL_BASE / 2.0
        ) / self.WHEEL_RADIUS

        right_speed = (
            linear
            + angular * self.WHEEL_BASE / 2.0
        ) / self.WHEEL_RADIUS

        p.setJointMotorControl2(
            self.robot_id,
            self.left_wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=left_speed,
            force=2.0,
            physicsClientId=self.client,
        )

        p.setJointMotorControl2(
            self.robot_id,
            self.right_wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=right_speed,
            force=2.0,
            physicsClientId=self.client,
        )

        # One RL action corresponds to eight PyBullet physics steps.
        for _ in range(8):
            p.stepSimulation(
                physicsClientId=self.client
            )

            self._update_dynamic_obstacles()

            if self.gui:
                time.sleep(
                    self.TIME_STEP
                )

        # --------------------------------------------------------------
        # Periodic global replanning.
        # --------------------------------------------------------------

        path_replanned = False

        if (
            self.planner_counter
            % PLANNER_UPDATE_FREQ
            == 0
        ):
            path_replanned = (
                self._plan_global_path()
            )

        # --------------------------------------------------------------
        # Observe new state.
        # --------------------------------------------------------------

        robot_pose = self._get_robot_pose()
        v, w = self.get_robot_velocity()
        lidar = self.get_lidar()

        # --------------------------------------------------------------
        # Update adaptive local waypoint.
        # --------------------------------------------------------------

        self._update_waypoint(
            robot_pose,
            v,
            lidar,
        )

        # --------------------------------------------------------------
        # Current path progress.
        # --------------------------------------------------------------

        current_path_progress = (
            self._get_current_path_progress(
                robot_pose
            )
        )

        # --------------------------------------------------------------
        # Calculate hybrid progress.
        #
        # IMPORTANT:
        #
        # If a new RRT path has just been accepted, its arc-length
        # coordinate starts at approximately zero because the new
        # path begins at the robot's current position.
        #
        # Therefore:
        #
        #     DO NOT calculate
        #
        #         new_s - old_s
        #
        # across a replan.
        #
        # Instead, re-anchor the reference and assign zero progress
        # to this transition.
        # --------------------------------------------------------------

        if path_replanned:

            if current_path_progress is not None:
                self.prev_path_progress = (
                    current_path_progress
                )

            progress_delta = 0.0

        elif (
            current_path_progress is not None
            and self.prev_path_progress is not None
        ):

            progress_delta = (
                current_path_progress
                - self.prev_path_progress
            )

            self.prev_path_progress = (
                current_path_progress
            )

        elif current_path_progress is not None:

            # A path has become available after previously having no
            # valid path. Anchor the reference without rewarding the
            # transition itself.
            self.prev_path_progress = (
                current_path_progress
            )

            progress_delta = 0.0

        else:

            # No valid RRT path is available.
            #
            # Use goal-distance progress as a fallback so that the
            # hybrid controller does not lose all dense progress
            # shaping during a temporary RRT failure.
            current_goal_distance = (
                self._distance_to_goal()
            )

            if self.prev_dist_to_wp is None:
                self.prev_dist_to_wp = (
                    current_goal_distance
                )

            progress_delta = (
                self.prev_dist_to_wp
                - current_goal_distance
            )

            self.prev_dist_to_wp = (
                current_goal_distance
            )

        # --------------------------------------------------------------
        # Local waypoint distance.
        #
        # This remains diagnostic only for the hybrid reward.
        # It is NOT used as the hybrid progress signal.
        # --------------------------------------------------------------

        curr_dist_to_wp = (
            self._waypoint_distance(
                robot_pose
            )
        )

        # --------------------------------------------------------------
        # Terminal conditions.
        # --------------------------------------------------------------

        collision = bool(
            self._check_collision()
        )

        safety_violation = bool(
            np.min(lidar)
            * self.LIDAR_RANGE
            < self.COLLISION_DIST
        )

        reached_goal = bool(
            self._check_goal()
        )

        xmin, xmax, ymin, ymax = (
            self.world_bounds
        )

        x, y = robot_pose[:2]

        out_of_bounds = not (
            xmin <= x <= xmax
            and ymin <= y <= ymax
        )

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------
        #
        # Hybrid progress is now path advancement:
        #
        #     Δs = s_t - s_(t-1)
        #
        # The adaptive waypoint is still part of the observation,
        # not the progress definition.
        # --------------------------------------------------------------

        reward, done = self.reward_calc.compute(
            prev_wp_dist=(
                self.prev_dist_to_wp
                if self.prev_dist_to_wp is not None
                else curr_dist_to_wp
            ),

            curr_wp_dist=curr_dist_to_wp,

            robot_pose=robot_pose,

            path=self.path,

            collision=collision,

            reached_goal=reached_goal,

            action=action,

            lidar_scan=lidar,

            goal_xy=self.goal_pos[:2],

            progress_delta=progress_delta,
        )

        # Keep local waypoint distance for diagnostics/compatibility.
        self.prev_dist_to_wp = (
            curr_dist_to_wp
        )

        # Out-of-bounds terminates the episode.
        #
        # It does not introduce an additional reward component.
        if out_of_bounds:
            done = True

        terminated = bool(done)

        truncated = bool(
            self.step_count >= MAX_STEPS
            and not terminated
        )

        # --------------------------------------------------------------
        # Diagnostic information.
        # --------------------------------------------------------------

        waypoint_robot = (
            self._waypoint_robot_frame(
                robot_pose
            )
        )

        obs = self._get_hybrid_observation()

        dynamic_positions = []
        dynamic_velocities = []

        for obstacle in getattr(
            self,
            "dynamic_obstacles",
            [],
        ):

            position = obstacle.get(
                "pos",
                [0.0, 0.0, 0.0],
            )

            velocity = obstacle.get(
                "vel",
                [0.0, 0.0, 0.0],
            )

            dynamic_positions.append(
                [
                    float(position[0]),
                    float(position[1]),
                ]
            )

            dynamic_velocities.append(
                [
                    float(velocity[0]),
                    float(velocity[1]),
                ]
            )

        # Exact reward-component audit data. This is instrumentation only;
        # it exposes the components already computed by RewardCalculator.
        reward_components = self.reward_calc.get_last_components()
        if reward_components is None:
            raise RuntimeError(
                "RewardCalculator did not provide last_components after compute()."
            )

        info = {
            "collision": collision,
            "reward_components": reward_components,

            "safety_violation": (
                safety_violation
            ),

            "reached_goal": reached_goal,

            "distance_to_goal": float(
                np.linalg.norm(
                    robot_pose[:2]
                    - self.goal_pos[:2]
                )
            ),

            # Diagnostic only.
            "waypoint_distance": float(
                curr_dist_to_wp
            ),

            "waypoint_x_robot": float(
                waypoint_robot[0]
            ),

            "waypoint_y_robot": float(
                waypoint_robot[1]
            ),

            "lookahead_distance": float(
                self.current_lookahead
            ),

            "path_available": (
                self.path is not None
            ),

            "path_index": int(
                self.path_index
            ),

            # New reward diagnostics.
            "path_progress": (
                float(current_path_progress)
                if current_path_progress is not None
                else None
            ),

            "path_progress_delta": float(
                progress_delta
            ),

            "path_replanned": bool(
                path_replanned
            ),

            "progress_mode": (
                "rrt_path_arc"
                if current_path_progress is not None
                else "goal_fallback"
            ),

            "out_of_bounds": bool(
                out_of_bounds
            ),

            "dynamic_obstacle_positions": (
                dynamic_positions
            ),

            "dynamic_obstacle_velocities": (
                dynamic_velocities
            ),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------
    # Global RRT planning
    # ------------------------------------------------------------------

    def _plan_global_path(self):
        """Plan a collision-free path through the static environment.

        Returns
        -------
        bool
            True when a new valid path is accepted.
            False when the existing path is retained or no valid path
            can be established.
        """

        if (
            self.robot_id is None
            or self.goal_pos is None
        ):
            return False

        start = np.asarray(
            self._get_robot_pose()[:2],
            dtype=np.float32,
        )

        goal = np.asarray(
            self.goal_pos[:2],
            dtype=np.float32,
        )

        # Use the public planner interface.
        self.planner.set_start(start)
        self.planner.set_goal(goal)

        new_path = self.planner.plan()

        # --------------------------------------------------------------
        # New RRT path accepted.
        # --------------------------------------------------------------

        if (
            new_path is not None
            and len(new_path) >= 2
        ):

            self.path = np.asarray(
                new_path,
                dtype=np.float32,
            )

            self.path_index = 0

            return True

        # --------------------------------------------------------------
        # RRT failure.
        #
        # If an existing valid path exists, preserve it.
        # --------------------------------------------------------------

        if (
            self.path is not None
            and len(self.path) >= 2
        ):
            return False

        # --------------------------------------------------------------
        # No previous path.
        #
        # Use direct path only if collision-free.
        # --------------------------------------------------------------

        if not self.planner.is_collision_segment(
            start,
            goal,
        ):

            self.path = np.vstack(
                [start, goal]
            ).astype(np.float32)

            self.path_index = 0

            return True

        # --------------------------------------------------------------
        # No valid path available.
        # --------------------------------------------------------------

        self.path = None
        self.path_index = 0

        return False

    # ------------------------------------------------------------------
    # Path progress
    # ------------------------------------------------------------------

    def _get_current_path_progress(
        self,
        robot_pose,
    ):
        """Return robot arc-length coordinate on the current RRT path.

        Returns None if no valid path is currently available.
        """

        if (
            self.path is None
            or len(self.path) < 2
        ):
            return None

        projection = (
            self._project_robot_onto_path(
                np.asarray(
                    robot_pose[:2],
                    dtype=np.float32,
                )
            )
        )

        if projection is None:
            return None

        segment_index, _, _, robot_arc = (
            projection
        )

        # Keep path following monotonic with respect to already passed
        # path segments.
        self.path_index = max(
            int(self.path_index),
            int(segment_index),
        )

        return float(robot_arc)

    # ------------------------------------------------------------------
    # Adaptive lookahead
    # ------------------------------------------------------------------

    def _compute_lookahead(
        self,
        linear_velocity,
        lidar,
    ):
        """Compute lookahead distance in meters.

        Modes
        -----
        fixed:
            Constant 0.30 m lookahead.

        velocity:
            Lookahead increases with forward velocity.

        velocity_clearance:
            Velocity-adaptive lookahead additionally modified by
            LiDAR clearance.
        """

        # --------------------------------------------------------------
        # Fixed baseline
        # --------------------------------------------------------------

        if self.lookahead_mode == "fixed":
            return 0.30

        # --------------------------------------------------------------
        # Velocity-adaptive component
        # --------------------------------------------------------------

        speed = max(
            0.0,
            float(linear_velocity),
        )

        lookahead = (
            0.30
            + 1.50 * speed
        )

        lookahead = float(
            np.clip(
                lookahead,
                0.20,
                0.80,
            )
        )

        # --------------------------------------------------------------
        # Velocity-only condition
        # --------------------------------------------------------------

        if self.lookahead_mode == "velocity":
            return lookahead

        # --------------------------------------------------------------
        # Velocity + clearance condition
        # --------------------------------------------------------------

        clearance = (
            float(np.min(lidar))
            * self.LIDAR_RANGE
        )

        if clearance < 0.40:
            lookahead *= 0.50

        elif clearance > 1.00:
            lookahead *= 1.20

        return float(
            np.clip(
                lookahead,
                0.20,
                0.80,
            )
        )


   
  
    @staticmethod
    def _segment_projection(
        point,
        a,
        b,
    ):
        """Project a point onto a line segment."""

        ab = b - a

        denom = float(
            np.dot(ab, ab)
        )

        if denom < 1e-12:
            return 0.0, a.copy()

        t = float(
            np.dot(
                point - a,
                ab,
            )
            / denom
        )

        t = float(
            np.clip(
                t,
                0.0,
                1.0,
            )
        )

        return (
            t,
            a + t * ab,
        )

    def _project_robot_onto_path(
        self,
        robot_xy,
    ):
        """Project robot onto the forward portion of the current path."""

        if (
            self.path is None
            or len(self.path) < 2
        ):
            return None

        robot_xy = np.asarray(
            robot_xy,
            dtype=np.float32,
        )

        start_segment = int(
            np.clip(
                self.path_index,
                0,
                len(self.path) - 2,
            )
        )

        prefix = 0.0

        for i in range(
            start_segment
        ):
            prefix += float(
                np.linalg.norm(
                    self.path[i + 1]
                    - self.path[i]
                )
            )

        best = None
        best_dist_sq = float(
            "inf"
        )

        running_arc = prefix

        for i in range(
            start_segment,
            len(self.path) - 1,
        ):

            a = self.path[i]
            b = self.path[i + 1]

            t, projection = (
                self._segment_projection(
                    robot_xy,
                    a,
                    b,
                )
            )

            dist_sq = float(
                np.sum(
                    (
                        robot_xy
                        - projection
                    )
                    ** 2
                )
            )

            segment_length = float(
                np.linalg.norm(
                    b - a
                )
            )

            if dist_sq < best_dist_sq:

                best_dist_sq = dist_sq

                best = (
                    i,
                    t,
                    projection.copy(),
                    running_arc
                    + t * segment_length,
                )

            running_arc += segment_length

        return best

    def _point_at_path_distance(
        self,
        arc_length,
    ):
        """Return the point at a specified arc distance along the path."""

        if (
            self.path is None
            or len(self.path) == 0
        ):
            return None

        if len(self.path) == 1:
            return self.path[0].copy()

        remaining = max(
            0.0,
            float(arc_length),
        )

        for i in range(
            len(self.path) - 1
        ):

            a = self.path[i]
            b = self.path[i + 1]

            seg_len = float(
                np.linalg.norm(
                    b - a
                )
            )

            if seg_len < 1e-12:
                continue

            if remaining <= seg_len:

                ratio = (
                    remaining
                    / seg_len
                )

                return (
                    a
                    + ratio * (b - a)
                ).astype(
                    np.float32
                )

            remaining -= seg_len

        return self.path[-1].copy()

    def _path_total_length(self):
        """Return total path arc length."""

        if (
            self.path is None
            or len(self.path) < 2
        ):
            return 0.0

        return float(
            np.sum(
                np.linalg.norm(
                    self.path[1:]
                    - self.path[:-1],
                    axis=1,
                )
            )
        )

    def _update_waypoint(
        self,
        robot_pose,
        linear_velocity,
        lidar,
    ):
        """Select an adaptive waypoint along the RRT path."""

        robot_xy = np.asarray(
            robot_pose[:2],
            dtype=np.float32,
        )

        self.current_lookahead = (
            self._compute_lookahead(
                linear_velocity,
                lidar,
            )
        )

        # --------------------------------------------------------------
        # No path: goal fallback.
        # --------------------------------------------------------------

        if (
            self.path is None
            or len(self.path) < 2
        ):

            self.current_waypoint = (
                np.asarray(
                    self.goal_pos[:2],
                    dtype=np.float32,
                )
            )

            return

        # --------------------------------------------------------------
        # Project robot onto path.
        # --------------------------------------------------------------

        projection = (
            self._project_robot_onto_path(
                robot_xy
            )
        )

        if projection is None:

            self.current_waypoint = (
                np.asarray(
                    self.goal_pos[:2],
                    dtype=np.float32,
                )
            )

            return

        (
            segment_index,
            _,
            _,
            robot_arc,
        ) = projection

        self.path_index = max(
            int(self.path_index),
            int(segment_index),
        )

        # --------------------------------------------------------------
        # Adaptive lookahead.
        # --------------------------------------------------------------

        total_length = (
            self._path_total_length()
        )

        target_arc = min(
            robot_arc
            + self.current_lookahead,
            total_length,
        )

        waypoint = (
            self._point_at_path_distance(
                target_arc
            )
        )

        if waypoint is None:

            self.current_waypoint = (
                np.asarray(
                    self.goal_pos[:2],
                    dtype=np.float32,
                )
            )

        else:

            self.current_waypoint = (
                np.asarray(
                    waypoint,
                    dtype=np.float32,
                )
            )

        # At the end of the global path, explicitly use the global goal.
        if (
            target_arc
            >= total_length - 1e-3
        ):

            self.current_waypoint = (
                np.asarray(
                    self.goal_pos[:2],
                    dtype=np.float32,
                )
            )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _waypoint_robot_frame(
        self,
        robot_pose,
    ):
        """Express the waypoint in the robot coordinate frame."""

        waypoint = (
            np.asarray(
                self.current_waypoint,
                dtype=np.float32,
            )
            if self.current_waypoint is not None
            else np.asarray(
                self.goal_pos[:2],
                dtype=np.float32,
            )
        )

        x, y, yaw = robot_pose

        dx = float(
            waypoint[0] - x
        )

        dy = float(
            waypoint[1] - y
        )

        c = np.cos(yaw)
        s = np.sin(yaw)

        dx_robot = (
            c * dx
            + s * dy
        )

        dy_robot = (
            -s * dx
            + c * dy
        )

        return np.array(
            [
                dx_robot,
                dy_robot,
                np.hypot(
                    dx_robot,
                    dy_robot,
                ),
            ],
            dtype=np.float32,
        )

    def _get_hybrid_observation(self):
        """Return 41D hybrid observation in physical units.

        Observation:
            36 LiDAR values
            + linear velocity v [m/s]
            + angular velocity w [rad/s]
            + waypoint dx [m] in robot frame
            + waypoint dy [m] in robot frame
            + waypoint distance [m]

        Numerical normalization is handled by VecNormalize during training.
        """

        lidar = self.get_lidar()
        v, w = self.get_robot_velocity()

        waypoint = self._waypoint_robot_frame(
            self._get_robot_pose()
        )

        obs = np.concatenate(
            [
                lidar.astype(np.float32),

                np.array(
                    [
                        v,
                        w,
                        waypoint[0],
                        waypoint[1],
                        waypoint[2],
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        return obs.astype(np.float32)
    
    

    def _waypoint_distance(
        self,
        robot_pose,
    ):
        """Distance from robot to current local waypoint."""

        target = (
            self.current_waypoint
            if self.current_waypoint is not None
            else self.goal_pos[:2]
        )

        return float(
            np.linalg.norm(
                np.asarray(
                    robot_pose[:2]
                )
                - np.asarray(target)
            )
        )