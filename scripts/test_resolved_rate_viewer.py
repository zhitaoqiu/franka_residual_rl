import time
import numpy as np
import mujoco
from mujoco import viewer

from controllers.pinocchio_model import PinocchioModel
from controllers.resolved_rate_ctrl import ResolvedRateController

MUJOCO_MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"
URDF_PATH = "assets/franka_panda/robots/panda_arm.urdf"

model = mujoco.MjModel.from_xml_path(MUJOCO_MODEL_PATH)
data = mujoco.MjData(model)

pin_model = PinocchioModel(URDF_PATH, ee_frame_name="panda_link8")
controller = ResolvedRateController(kp=1.0, dq_limit=0.03)

mujoco.mj_forward(model, data)

target_pos = np.array([0.30, 0.00, 0.70], dtype=np.float64)

print("initial qpos:", data.qpos[:7])

with viewer.launch_passive(model, data) as v:
    for step in range(2000):
        q_current = data.qpos[:7].copy()

        ee_pos_pin = pin_model.get_ee_position(q_current)
        J = pin_model.get_frame_jacobian(q_current)

        q_target = controller.compute_q_target(
            q_current=q_current,
            ee_pos=ee_pos_pin,
            target_pos=target_pos,
            J_full=J,
        )

        ctrl = np.zeros(model.nu)
        ctrl[:7] = q_target
        if model.nu > 7:
            ctrl[7] = 0.0

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if step % 50 == 0:
            dist = np.linalg.norm(target_pos - ee_pos_pin)
            print(f"step={step:04d}, ee={ee_pos_pin}, dist={dist:.4f}")

        v.sync()
        time.sleep(0.01)