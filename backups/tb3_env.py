import gym
import numpy as np
import pybullet as p
import pybullet_data
import random
import time
from gym import spaces

from config import (
    TIME_STEP, MAX_STEPS, NUM_LIDAR, LIDAR_RANGE, LIDAR_OFFSET,
    GOAL_THRESHOLD, COLLISION_DIST, LINEAR_VEL, ANGULAR_VEL,
    WHEEL_RADIUS, WHEEL_BASE, STATIC_OBSTACLES, DYNAMIC_CONFIG, ENV_TYPE,
    GUI
)

class TB3Env(gym.Env):
    """Base TurtleBot3 environment – Gymnasium compatible (5‑value step)."""

    def __init__(self, env_type=ENV_TYPE, gui=GUI):
        super().__init__()
        self.env_type = env_type
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Robot handles
        self.robot_id = None
        self.left_wheel = None
        self.right_wheel = None
        self.goal_id = None
        self.static_obstacles = []
        self.dynamic_obstacles = []

        # World bounds
        self.world_bounds = [-3.0, 3.0, -3.0, 3.0]

        # State
        self.step_count = 0
        self.prev_dist_to_goal = None
        self.goal_pos = np.array([1.5, 1.5, 0.0])

        # Build world (will be fully reset in reset)
        self._load_world()
        self._load_robot()
        self._place_goal()
        self._spawn_static_obstacles()
        self._init_dynamic_obstacles()

        # Observation space: LiDAR (36) + v,w (2) + dx,dy,dist (3) = 41
        self.obs_dim = NUM_LIDAR + 2 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.LIDAR_RANGE = LIDAR_RANGE

    # ------------------------------------------------------------------
    # Setup methods (unchanged, same as before)
    # ------------------------------------------------------------------
    def _load_world(self):
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=TIME_STEP, numSubSteps=1, numSolverIterations=10)
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

    def _load_robot(self):
        start_pos = [0.0, 0.0, 0.0]
        start_ori = [0, 0, 0, 1]
        self.robot_id = p.loadURDF(
            "assets/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
            start_pos, start_ori
        )
        for i in range(p.getNumJoints(self.robot_id)):
            name = p.getJointInfo(self.robot_id, i)[1].decode()
            if "wheel_left" in name:
                self.left_wheel = i
            elif "wheel_right" in name:
                self.right_wheel = i
        assert self.left_wheel is not None and self.right_wheel is not None

    def _place_goal(self):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0,1,0,1])
        self.goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=self.goal_pos.tolist())

    def _spawn_static_obstacles(self):
        self.static_obstacles = []
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.1])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.1], rgbaColor=[1,0,0,1])
        for x, y in STATIC_OBSTACLES:
            body = p.createMultiBody(baseMass=0,
                                     baseCollisionShapeIndex=box_collision,
                                     baseVisualShapeIndex=box_visual,
                                     basePosition=[x, y, 0.1])
            self.static_obstacles.append(body)

    def _init_dynamic_obstacles(self):
        self.dynamic_obstacles = []
        cfg = DYNAMIC_CONFIG.get(self.env_type, {'num':0})
        for _ in range(cfg.get('num', 0)):
            x = random.uniform(-2.5, 2.5)
            y = random.uniform(-2.5, 2.5)
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0,0,1,1])
            body = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col,
                                     baseVisualShapeIndex=vis, basePosition=[x,y,0.1])
            vel = [random.uniform(-cfg.get('speed',0.2), cfg.get('speed',0.2)),
                   random.uniform(-cfg.get('speed',0.2), cfg.get('speed',0.2)), 0]
            self.dynamic_obstacles.append({'id': body, 'pos': [x,y,0.1], 'vel': vel})

    def _update_dynamic_obstacles(self):
        cfg = DYNAMIC_CONFIG.get(self.env_type, {'num':0})
        bounds = cfg.get('bounds', [-3,3,-3,3])
        for obs in self.dynamic_obstacles:
            obs['pos'][0] += obs['vel'][0] * TIME_STEP
            obs['pos'][1] += obs['vel'][1] * TIME_STEP
            if obs['pos'][0] < bounds[0] or obs['pos'][0] > bounds[1]:
                obs['vel'][0] *= -1
            if obs['pos'][1] < bounds[2] or obs['pos'][1] > bounds[3]:
                obs['vel'][1] *= -1
            p.resetBasePositionAndOrientation(obs['id'], obs['pos'], [0,0,0,1])

    # ------------------------------------------------------------------
    # Core Gym API (Gymnasium style)
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Reset the environment. Returns (obs, info)."""
        # Optionally set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.step_count = 0
        self._load_world()
        self._load_robot()
        # Random goal (avoid too close to start)
        while True:
            gx = random.uniform(-2.5, 2.5)
            gy = random.uniform(-2.5, 2.5)
            if np.hypot(gx, gy) > 0.8:
                self.goal_pos = np.array([gx, gy, 0.0])
                break
        self._place_goal()
        self._spawn_static_obstacles()
        self._init_dynamic_obstacles()

        # Random start pose
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)
        yaw = random.uniform(-np.pi, np.pi)
        p.resetBasePositionAndOrientation(self.robot_id, [x,y,0.0],
                                          p.getQuaternionFromEuler([0,0,yaw]))

        # Settling loop
        for _ in range(240):
            p.stepSimulation()
            if self.gui:
                time.sleep(TIME_STEP)

        self.prev_dist_to_goal = self._distance_to_goal()
        obs = self._get_observation()
        info = {}   # no additional info
        return obs, info

    def step(self, action):
        """Take an action. Returns (obs, reward, terminated, truncated, info)."""
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0)

        # Wheel velocity control
        linear = action[0] * LINEAR_VEL
        angular = action[1] * ANGULAR_VEL
        left = (linear - (WHEEL_BASE/2.0)*angular) / WHEEL_RADIUS
        right = (linear + (WHEEL_BASE/2.0)*angular) / WHEEL_RADIUS
        p.setJointMotorControl2(self.robot_id, self.left_wheel, p.VELOCITY_CONTROL,
                                targetVelocity=left, force=2.0)
        p.setJointMotorControl2(self.robot_id, self.right_wheel, p.VELOCITY_CONTROL,
                                targetVelocity=right, force=2.0)

        # Physics step (8 sub‑steps)
        for _ in range(8):
            p.stepSimulation()
            self._update_dynamic_obstacles()
            if self.gui:
                time.sleep(TIME_STEP)

        obs = self._get_observation()
        reward, done = self._compute_reward(action)

        robot_pose = self._get_robot_pose()
        x, y = robot_pose[0], robot_pose[1]
        if x < -3.5 or x > 3.5 or y < -3.5 or y > 3.5:
            reward -= 2.0          # penalty for leaving the world
            print("Out of bounds! reward -2.0")
            done = True            # terminate episode

        # Gymnasium: terminated = episode ended (goal/collision), truncated = max steps
        terminated = done and not (self.step_count >= MAX_STEPS)
        truncated = (self.step_count >= MAX_STEPS) and not done
        info = {"collision": self._check_collision(), "reached_goal": self._check_goal()}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper methods (unchanged)
    # ------------------------------------------------------------------
    def _get_observation(self):
        lidar = self._get_lidar()
        v, w = self._get_robot_velocity()
        dx = self.goal_pos[0] - self._get_robot_pose()[0]
        dy = self.goal_pos[1] - self._get_robot_pose()[1]
        dist = np.hypot(dx, dy)
        obs = np.concatenate([lidar, [v, w], [dx, dy, dist]]).astype(np.float32)
        return obs

    def _get_lidar(self):
        from env.lidar import get_lidar_scan
        return get_lidar_scan(self.robot_id, NUM_LIDAR, LIDAR_RANGE, LIDAR_OFFSET)

    def _get_robot_pose(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = p.getEulerFromQuaternion(ori)
        return np.array([pos[0], pos[1], yaw])

    def _get_robot_velocity(self):
        lin, ang = p.getBaseVelocity(self.robot_id)
        v = np.hypot(lin[0], lin[1])
        return v, ang[2]

    def _distance_to_goal(self):
        pose = self._get_robot_pose()
        return np.hypot(self.goal_pos[0]-pose[0], self.goal_pos[1]-pose[1])

    def _check_collision(self):
        lidar = self._get_lidar()
        return np.min(lidar) * LIDAR_RANGE < COLLISION_DIST

    def _check_goal(self):
        return self._distance_to_goal() < GOAL_THRESHOLD

   # ---------------- REWARD ----------------
    def _compute_reward(self, action):
        curr_dist = self._distance_to_goal()

        # CLAMPED PROGRESS
        progress = max(0.0, self.prev_dist_to_goal - curr_dist)

        reward = 6.0 * progress
        reward += 0.2 * max(0, action[0])
        reward -= 0.05 * abs(action[1])
        reward -= 0.02  # time penalty

        done = False

        if self._check_collision():
            reward -= 150
            print("Collision! reward -150")
            done = True

        elif self._check_goal():
            reward += 300
            print("Goal reached! +300")
            done = True
        elif self.step_count >= MAX_STEPS:
            print(f"Max steps reached: {self.step_count} >= {MAX_STEPS}")
            done = True

        self.prev_dist_to_goal = curr_dist

        return reward, done
   
      
    # ------------------------------------------------------------------
    # Public methods (unchanged)
    # ------------------------------------------------------------------
    def get_robot_pose(self):
        return self._get_robot_pose()

    def get_robot_velocity(self):
        return self._get_robot_velocity()

    def get_lidar(self):
        return self._get_lidar()

    def get_dynamic_obstacle_distances(self):
        robot_xy = self.get_robot_pose()[:2]
        dists = []
        for obs in self.dynamic_obstacles:
            ox, oy = obs['pos'][0], obs['pos'][1]
            dists.append(np.hypot(robot_xy[0]-ox, robot_xy[1]-oy))
        return dists

    def get_robot_position(self):
        return self.get_robot_pose()[:2]

    def get_obs(self):
        return self.get_lidar()
    
    def draw_path(self, path, color=[0,1,0]):
        """Draw RRT path as connected lines in XY plane."""
        if path is None or len(path) < 2:
            return
        z = 0.05
        for i in range(len(path)-1):
            p0 = [path[i][0], path[i][1], z]
            p1 = [path[i+1][0], path[i+1][1], z]
            p.addUserDebugLine(p0, p1, lineColorRGB=color, lineWidth=2.0)