import mujoco

MODEL_PATH = "assets/franka_panda/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("=== basic info ===")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)
print()

print("=== actuators ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    trnid0 = model.actuator_trnid[i, 0]
    trnid1 = model.actuator_trnid[i, 1]

    joint_name = None
    if trnid0 >= 0:
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid0)

    print(f"actuator[{i}]")
    print("  name      =", name)
    print("  joint     =", joint_name)
    print("  trnid     =", (int(trnid0), int(trnid1)))
    print("  ctrlrange =", model.actuator_ctrlrange[i])
    print("  gear      =", model.actuator_gear[i])
    print()