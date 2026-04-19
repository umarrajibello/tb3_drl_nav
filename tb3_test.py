import pybullet as p
import pybullet_data
import time

# 1) Connect and basic world
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")

# 2) Load TurtleBot3 a bit above the ground (it will drop and settle)
robot = p.loadURDF(
    "assets/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.urdf",
    basePosition=[0, 0, 0.0],
    baseOrientation=[0, 0, 0, 1]
)

# Let it settle
for _ in range(240):
    p.stepSimulation()
    time.sleep(1/240)

print("CONTACT:", p.getContactPoints(robot, plane_id))

# 3) Find wheel joints
left = None
right = None
for i in range(p.getNumJoints(robot)):
    name = p.getJointInfo(robot, i)[1].decode()
    print(i, name)
    if "wheel_left" in name:
        left = i
    if "wheel_right" in name:
        right = i

print("Left wheel joint:", left)
print("Right wheel joint:", right)

# 4) Drive forward and print pose
for step in range(500):
    p.setJointMotorControl2(robot, left,  p.VELOCITY_CONTROL,
                            targetVelocity=10, force=2.0)
    p.setJointMotorControl2(robot, right, p.VELOCITY_CONTROL,
                            targetVelocity=10, force=2.0)

    p.stepSimulation()
    pos, orn = p.getBasePositionAndOrientation(robot)
    print(step, pos)
    time.sleep(1/240)

# 5) Keep GUI open so you can look around
print("Done driving. Press Ctrl+C in the terminal to quit.")
while True:
    p.stepSimulation()
    time.sleep(1/240)
