"""Compare MuJoCo FK vs Pinocchio FK for cross-validation."""
import mujoco
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from controllers.pinocchio_model import PinocchioModel


def main():
    xml_path = os.path.join(
        project_root, "assets", "franka_panda", "franka_emika_panda", "scene.xml"
    )
    urdf_path = os.path.join(
        project_root, "assets", "franka_panda", "robots", "panda_arm.urdf"
    )

    print(f"Loading MuJoCo XML: {xml_path}")
    print(f"Loading URDF: {urdf_path}")

    try:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
    except Exception as e:
        print(f"MuJoCo model load failed: {e}")
        return

    try:
        pin_model = PinocchioModel(urdf_path, ee_frame_name="panda_link8")
    except Exception as e:
        print(f"Pinocchio model load failed: {e}")
        return

    q_test = np.array([0.5, -0.3, 0.2, -1.5, 0.4, 1.2, 0.3])
    dq_test = np.zeros(7)

    mj_data.qpos[:7] = q_test
    mujoco.mj_kinematics(mj_model, mj_data)

    ee_id_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    if ee_id_mj == -1:
        ee_id_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if ee_id_mj == -1:
        print("ERROR: cannot find end-effector body in MuJoCo model")
        return

    ee_pos_mj = mj_data.xpos[ee_id_mj]

    pin_model.update_state(q_test, dq_test)
    ee_pos_pin, _ = pin_model.get_ee_pose()

    print("\n" + "=" * 50)
    print("[FK Cross-Validation]")
    print(f"MuJoCo    ee pos (xyz): {ee_pos_mj}")
    print(f"Pinocchio ee pos (xyz): {ee_pos_pin}")

    error = np.linalg.norm(ee_pos_mj - ee_pos_pin)
    print(f"Absolute position error: {error:.8f} m")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
