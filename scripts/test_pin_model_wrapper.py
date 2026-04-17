import numpy as np
from controllers.pinocchio_model import PinocchioModel

URDF_PATH = "assets/franka_panda/robots/panda_arm.urdf"

robot = PinocchioModel(URDF_PATH, ee_frame_name="panda_link8")

q = np.zeros(7)
ee_pos = robot.get_ee_position(q)
J = robot.get_frame_jacobian(q)

print("ee_pos:", ee_pos)
print("J shape:", J.shape)
print(J)