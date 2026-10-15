import numpy as np
import random

class RRTPlanner:
    def __init__(self, static_obstacles, bounds=[-3, 3, -3, 3], step_size=0.2, max_iter=1000):
        self.obstacles = static_obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.start = None
        self.goal = None
        self.path = None

    def set_start(self, start_xy):
        self.start = np.array(start_xy[:2])

    def set_goal(self, goal_xy):
        self.goal = np.array(goal_xy[:2])

    def is_collision(self, p):
        for obs in self.obstacles:
            if np.linalg.norm(p - np.array(obs[:2])) < 0.35:
                return True
        if p[0] < self.bounds[0] or p[0] > self.bounds[1] or p[1] < self.bounds[2] or p[1] > self.bounds[3]:
            return True
        return False

    def nearest(self, sample):
        distances = [np.linalg.norm(node - sample) for node in self.nodes]
        return np.argmin(distances)

    def steer(self, from_node, to_node):
        dir_vec = to_node - from_node
        dist = np.linalg.norm(dir_vec)
        if dist < self.step_size:
            return to_node
        return from_node + (dir_vec / dist) * self.step_size

    def plan(self):
        if self.start is None or self.goal is None:
            return None
        self.nodes = [self.start]
        self.parents = [-1]

        for _ in range(self.max_iter):
            if random.random() < 0.05:
                sample = self.goal
            else:
                sample = np.array([
                    random.uniform(self.bounds[0], self.bounds[1]),
                    random.uniform(self.bounds[2], self.bounds[3])
                ])
            nearest_idx = self.nearest(sample)
            new_node = self.steer(self.nodes[nearest_idx], sample)
            if not self.is_collision(new_node):
                self.nodes.append(new_node)
                self.parents.append(nearest_idx)
                if np.linalg.norm(new_node - self.goal) < self.step_size:
                    path = [self.goal]
                    idx = len(self.nodes)-1
                    while idx != -1:
                        path.append(self.nodes[idx])
                        idx = self.parents[idx]
                    self.path = list(reversed(path))
                    return self.path
        return None

    def get_waypoint(self, distance=0.5):
        if self.path is None:
            return self.goal
        cum_dist = 0
        for i in range(len(self.path)-1):
            seg_len = np.linalg.norm(self.path[i+1] - self.path[i])
            if cum_dist + seg_len >= distance:
                ratio = (distance - cum_dist) / seg_len
                waypoint = self.path[i] + ratio * (self.path[i+1] - self.path[i])
                return waypoint
            cum_dist += seg_len
        return self.goal