import numpy as np
from envs.mujoco_base import MujocoBase
from controllers.pinocchio_model import PinocchioModel
from controllers.resolved_rate_ctrl import ResolvedRateController

MUJOCO_MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"
URDF_PATH = "assets/franka_panda/robots/panda_arm.urdf"

env = MujocoBase(MUJOCO_MODEL_PATH, ee_body_name="panda_hand")
pin_model = PinocchioModel(URDF_PATH, ee_frame_name="panda_link8")
controller = ResolvedRateController(kp=1.0, dq_limit=0.03)

target_pos = np.array([0.35, 0.0, 0.55], dtype=np.float64)

obs = env.reset()
print("reset ok")
print("initial mujoco ee_pos:", obs["ee_pos"])
print("target_pos:", target_pos)

for i in range(50):
    q_current = obs["qpos"][:7].copy()

    ee_pos_pin = pin_model.get_ee_position(q_current)
    J = pin_model.get_frame_jacobian(q_current)

    q_target = controller.compute_q_target(
        q_current=q_current,
        ee_pos=ee_pos_pin,
        target_pos=target_pos,
        J_full=J,
    )

    ctrl = np.zeros(env.nu)
    ctrl[:7] = q_target
    if env.nu > 7:
        ctrl[7] = 0.0

    env.step_sim(ctrl, n_substeps=5)
    obs = env.get_obs()

    dist = np.linalg.norm(target_pos - ee_pos_pin)

    print(
        f"step={i:02d}, "
        f"pin_ee={ee_pos_pin}, "
        f"dist={dist:.4f}"
    )