import time
import numpy as np
import mujoco
from mujoco import viewer

MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

# 初始状态
mujoco.mj_forward(model, data)
q0 = data.qpos.copy()
print("initial qpos:", q0)

# 只控制前7个关节，忽略第8个奇怪 actuator
target = q0.copy()
target[0] += 0.3
target[1] -= 0.2
target[3] += 0.2

with viewer.launch_passive(model, data) as v:
    for t in range(2000):
        ctrl = np.zeros(model.nu)

        # 假设前7维是关节位置目标
        ctrl[:7] = target[:7]

        # 第8维先不管
        if model.nu > 7:
            ctrl[7] = 0.0

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        if t % 100 == 0:
            print("step", t, "qpos[:7] =", data.qpos[:7])

        v.sync()
        time.sleep(0.01)