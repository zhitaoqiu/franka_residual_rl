import time
import numpy as np
import mujoco
from mujoco import viewer
from controllers.joint_position_ctrl import JointPositionController

MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)
q0 = data.qpos.copy()

q_target = q0.copy()
q_target[0] += 0.3
q_target[1] -= 0.2
q_target[3] += 0.2

controller = JointPositionController(kp=0.2, ctrl_limit=0.1)

with viewer.launch_passive(model, data) as v:
    for t in range(2000):
        q_current = data.qpos[:7].copy()
        ctrl7 = controller.compute(q_current, q_target[:7])

        ctrl = np.zeros(model.nu)
        ctrl[:7] = ctrl7
        if model.nu > 7:
            ctrl[7] = 0.0

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if t % 100 == 0:
            err = np.linalg.norm(q_target[:7] - data.qpos[:7])
            print(f"step={t}, joint_err={err:.4f}")

        v.sync()
        time.sleep(0.01)