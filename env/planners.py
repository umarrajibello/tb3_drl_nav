# env/planners.py
import random
import numpy as np


class RRTPlanner:
    """RRT planner for static obstacles.

    Public interface is preserved for the existing hybrid environment.
    Dynamic obstacles are intentionally handled by the local policy.
    """

    def __init__(
        self,
        static_obstacles,
        bounds=[-3, 3, -3, 3],
        step_size=0.2,
        max_iter=1000,
        obstacle_half_extent=0.2,
        robot_clearance=0.15,
        edge_resolution=0.05,
    ):
        self.obstacles = [
            np.asarray(obs[:2], dtype=np.float32)
            for obs in static_obstacles
        ]
        self.bounds = list(bounds)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.obstacle_half_extent = float(obstacle_half_extent)
        self.robot_clearance = float(robot_clearance)
        self.edge_resolution = float(edge_resolution)

        self.start = None
        self.goal = None
        self.path = None
        self.nodes = []
        self.parents = []

    def set_start(self, start_xy):
        self.start = np.asarray(start_xy[:2], dtype=np.float32)

    def set_goal(self, goal_xy):
        self.goal = np.asarray(goal_xy[:2], dtype=np.float32)

    def _in_bounds(self, point):
        return bool(
            self.bounds[0] <= point[0] <= self.bounds[1]
            and self.bounds[2] <= point[1] <= self.bounds[3]
        )

    def _obstacle_contains(self, point, clearance=None):
        """Check the XY footprint of the simulated 0.4 m square obstacle."""
        if clearance is None:
            clearance = self.robot_clearance

        half_extent = self.obstacle_half_extent + float(clearance)

        for obs in self.obstacles:
            if (
                abs(float(point[0]) - float(obs[0])) <= half_extent
                and abs(float(point[1]) - float(obs[1])) <= half_extent
            ):
                return True
        return False

    def is_collision(self, point):
        point = np.asarray(point[:2], dtype=np.float32)
        return (not self._in_bounds(point)) or self._obstacle_contains(point)

    def is_collision_segment(self, p1, p2, resolution=None):
        """Check the complete edge rather than only its endpoints."""
        p1 = np.asarray(p1[:2], dtype=np.float32)
        p2 = np.asarray(p2[:2], dtype=np.float32)

        if resolution is None:
            resolution = self.edge_resolution

        distance = float(np.linalg.norm(p2 - p1))
        n = max(2, int(np.ceil(distance / resolution)) + 1)

        for t in np.linspace(0.0, 1.0, n):
            point = p1 + t * (p2 - p1)
            if self.is_collision(point):
                return True

        return False

    def nearest(self, sample):
        if not self.nodes:
            raise RuntimeError("RRT has no nodes.")

        sample = np.asarray(sample[:2], dtype=np.float32)
        distances = [np.linalg.norm(node - sample) for node in self.nodes]
        return int(np.argmin(distances))

    def steer(self, from_node, to_node):
        from_node = np.asarray(from_node[:2], dtype=np.float32)
        to_node = np.asarray(to_node[:2], dtype=np.float32)

        direction = to_node - from_node
        distance = float(np.linalg.norm(direction))

        if distance < 1e-8:
            return from_node.copy()
        if distance <= self.step_size:
            return to_node.copy()

        return from_node + direction / distance * self.step_size

    def _sample_free(self):
        return np.array(
            [
                random.uniform(self.bounds[0], self.bounds[1]),
                random.uniform(self.bounds[2], self.bounds[3]),
            ],
            dtype=np.float32,
        )

    def _validate_start_goal(self):
        if self.start is None or self.goal is None:
            return False
        return not self.is_collision(self.start) and not self.is_collision(self.goal)

    def _build_path(self, goal_parent_idx):
        path = [self.goal.copy()]
        idx = int(goal_parent_idx)

        while idx != -1:
            path.append(self.nodes[idx].copy())
            idx = self.parents[idx]

        path.reverse()
        return path

    def _shortcut_path(self, path):
        """Remove intermediate vertices when a direct safe connection exists."""
        if path is None or len(path) <= 2:
            return path

        simplified = [np.asarray(path[0], dtype=np.float32)]
        current = 0

        while current < len(path) - 1:
            next_idx = len(path) - 1

            while next_idx > current + 1:
                if not self.is_collision_segment(path[current], path[next_idx]):
                    break
                next_idx -= 1

            simplified.append(np.asarray(path[next_idx], dtype=np.float32))
            current = next_idx

        return simplified


    def plan(self):
        """Return a collision-free path or None."""
        self.path = None
        self.nodes = []
        self.parents = []

        if self.start is None or self.goal is None:
            return None

        # Normalize start and goal again at the planner boundary.
        # This protects against tuple/list inputs from the environment.
        self.start = np.asarray(self.start[:2], dtype=np.float32).copy()
        self.goal = np.asarray(self.goal[:2], dtype=np.float32).copy()

        if not self._validate_start_goal():
            return None
    
        # Fast path for unobstructed scenes.
        if not self.is_collision_segment(self.start, self.goal):
            self.path = [self.start.copy(), self.goal.copy()]
            return self.path

        self.nodes = [self.start.copy()]
        self.parents = [-1]

        for _ in range(self.max_iter):
            if random.random() < 0.05:
                sample = self.goal.copy()
            else:
                sample = self._sample_free()

            nearest_idx = self.nearest(sample)
            nearest_node = self.nodes[nearest_idx]
            new_node = self.steer(nearest_node, sample)

            # Critical correction: check the whole parent -> new_node edge.
            if self.is_collision_segment(nearest_node, new_node):
                continue

            self.nodes.append(new_node)
            self.parents.append(nearest_idx)
            new_idx = len(self.nodes) - 1

            # Critical correction: also check the final edge to the goal.
            if (
                np.linalg.norm(new_node - self.goal) <= self.step_size
                and not self.is_collision_segment(new_node, self.goal)
            ):
                raw_path = self._build_path(new_idx)
                self.path = self._shortcut_path(raw_path)
                return self.path

        self.path = None
        return None


    

    def get_waypoint(self, distance=0.5):
        """Return a point approximately distance meters along the path.

        Kept for compatibility. TB3HybridEnv currently performs its own
        adaptive waypoint selection.
        """
        if self.path is None or len(self.path) == 0:
            return self.goal

        distance = max(float(distance), 0.0)
        cumulative = 0.0

        for i in range(len(self.path) - 1):
            p0 = np.asarray(self.path[i], dtype=np.float32)
            p1 = np.asarray(self.path[i + 1], dtype=np.float32)
            segment = float(np.linalg.norm(p1 - p0))

            if segment < 1e-8:
                continue

            if cumulative + segment >= distance:
                ratio = (distance - cumulative) / segment
                return p0 + ratio * (p1 - p0)

            cumulative += segment

        return np.asarray(self.path[-1], dtype=np.float32)
