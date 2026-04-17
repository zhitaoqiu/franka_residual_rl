import mujoco
from mujoco import viewer

MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Model loaded successfully")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

with viewer.launch_passive(model, data) as v:
    for _ in range(10000):
        mujoco.mj_step(model, data)
        v.sync()