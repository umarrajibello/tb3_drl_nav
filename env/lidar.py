import pybullet as p
import numpy as np
from config import LIDAR_RANGE, LIDAR_OFFSET, NUM_LIDAR, LIDAR_NOISE_STD, LIDAR_DROPOUT_RATE

def get_lidar_scan(robot_id, physics_client_id, num_rays=NUM_LIDAR, max_distance=LIDAR_RANGE,
                   lidar_offset=LIDAR_OFFSET, visualize=False, ray_skip=2):
    """
    Simulate 2D LiDAR using ray casting.
    Returns normalized distances in [0,1] (0 = obstacle at robot, 1 = max range).
    Includes optional Gaussian noise and dropout for sim‑to‑real robustness.
    """
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id, physicsClientId=physics_client_id)
    yaw = p.getEulerFromQuaternion(base_orn)[2]

    # Angles evenly spaced around 360°, avoiding duplicate at +π
    angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
    height = 0.15  # LiDAR height above ground

    ray_from = []
    ray_to = []

    for angle in angles:
        ray_angle = yaw + angle
        fx = base_pos[0] + lidar_offset * np.cos(ray_angle)
        fy = base_pos[1] + lidar_offset * np.sin(ray_angle)
        fz = base_pos[2] + height

        tx = base_pos[0] + max_distance * np.cos(ray_angle)
        ty = base_pos[1] + max_distance * np.sin(ray_angle)
        tz = fz

        ray_from.append([fx, fy, fz])
        ray_to.append([tx, ty, tz])

    # Batch ray test
    results = p.rayTestBatch(ray_from, ray_to, physicsClientId=physics_client_id)

    distances = []
    for i, res in enumerate(results):
        hit_fraction = res[2]   # 0 = at start, 1 = at end
        hit_obj = res[0]
        dist = hit_fraction * max_distance
        # Ignore self hits
        if hit_obj == robot_id:
            dist = max_distance
        # Normalize to [0,1]
        norm = dist / max_distance
        distances.append(norm)

        # Optional visualization (debug)
        if visualize and i % ray_skip == 0:
            hit_point = res[3] if hit_fraction < 1.0 else ray_to[i]
            color = [1, 0, 0] if dist < 0.5 else [0, 1, 0]
            p.addUserDebugLine(ray_from[i], hit_point, color, lifeTime=0.05)

    distances = np.array(distances, dtype=np.float32)

    # ---- ADD NOISE (only if enabled in config) ----
    if LIDAR_NOISE_STD > 0:
        noise = np.random.normal(0, LIDAR_NOISE_STD, size=distances.shape)
        distances = np.clip(distances + noise, 0.0, 1.0)
    if LIDAR_DROPOUT_RATE > 0:
        dropout = np.random.random(distances.shape) < LIDAR_DROPOUT_RATE
        distances[dropout] = 1.0
    # -----------------------------------------------

    return distances