import numpy as np

# Simulation
GUI = True       # default, overridden by command line
TIME_STEP = 1./240.
MAX_STEPS = 1000
# Environment type (will be overridden by command line)
ENV_TYPE = 'static'   # default, can be 'static', 'mixed', or 'dense'

# RobotA
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.16
LINEAR_VEL = 0.22
ANGULAR_VEL = 2.84

# LiDAR
NUM_LIDAR = 36
LIDAR_RANGE = 3.5
LIDAR_OFFSET = 0.20

# Goal & Reward
GOAL_THRESHOLD = 0.2
COLLISION_DIST = 0.15
# Goal-distance curriculum
GOAL_DIST_MIN = 1.0
GOAL_DIST_MAX = 2.0

#STATIC_OBSTACLES = [ (-1.2, 0.8), (-1.5, -0.5), (0.5, -1.2)
                   # ]  #No obstacles for now.
# Obstacle configurations

STATIC_OBSTACLES = [ (-1.2, 0.8), (-1.5, -0.5), (0.5, -1.2),
     (1.2, 1.2), (1.8, 0.5), 
   (-1.5, -0.5),  (1.5, -0.8), (-0.5, 1.5)
]
DYNAMIC_CONFIG = {
    'static': {'num': 0},
    'mixed': {'num': 5, 'type': 'linear_bounce', 'speed': 0.2, 'bounds': [-3, 3, -3, 3]},
    'dense': {'num': 10, 'type': 'linear_bounce', 'speed': 0.2, 'bounds': [-3, 3, -3, 3]}
}

# Global planner (RRT)
PLANNER_TYPE = 'rrt'
PLANNER_UPDATE_FREQ = 50      # steps between replans
WAYPOINT_DISTANCE = 1.0       # meters ahead on the path

# Temporal observation (for LSTM and Transformer)
OBSERVATION_HISTORY = 8       # number of past LiDAR scans
# LiDAR noise parameters (set to zero for perfect sensor, old behaviour)
LIDAR_NOISE_STD = 0.0       # standard deviation of Gaussian noise (e.g., 0.02)
LIDAR_DROPOUT_RATE = 0.0    # fraction of rays that return max range (e.g., 0.03)