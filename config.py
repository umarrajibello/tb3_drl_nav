import numpy as np

# Simulation
GUI = False                 # default, overridden by command line
TIME_STEP = 1./240.
MAX_STEPS = 5000
# Environment type (will be overridden by command line)
ENV_TYPE = 'static'   # default, can be 'static', 'mixed', or 'dense'

# Robot
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
LINEAR_VEL = 0.22
ANGULAR_VEL = 2.84

# LiDAR
NUM_LIDAR = 36
LIDAR_RANGE = 3.5
LIDAR_OFFSET = 0.20

# Goal & Reward
GOAL_THRESHOLD = 0.3
COLLISION_DIST = 0.15

#STATIC_OBSTACLES = []  #No obstacles for now.
# Obstacle configurations

STATIC_OBSTACLES = [
     (1.2, 1.2), (1.8, 0.5), (0.5, 1.8), (-0.5, 1.5),
     (-1.2, 0.8), (-1.5, -0.5), (0.5, -1.2), (1.5, -0.8)
]
DYNAMIC_CONFIG = {
    'static': {'num': 0},
    'mixed': {'num': 5, 'type': 'linear_bounce', 'speed': 0.2, 'bounds': [-3, 3]},
    'dense': {'num': 10, 'type': 'linear_bounce', 'speed': 0.2, 'bounds': [-3, 3]}
}

# Global planner (RRT)
PLANNER_TYPE = 'rrt'
PLANNER_UPDATE_FREQ = 10      # steps between replans
WAYPOINT_DISTANCE = 0.5       # meters ahead on the path

# Temporal observation (for LSTM and Transformer)
OBSERVATION_HISTORY = 8       # number of past LiDAR scans