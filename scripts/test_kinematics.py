# filepath: scripts/test_kinematics.py
import mujoco
import numpy as np
import os
import sys

# 将上一级目录加入系统路径，以便导入 controllers
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from controllers.pinocchio_model import PinocchioModel

def main():
    # 1. 路径准备 (严格匹配你目前的文件夹结构)
    xml_path = os.path.join(project_root, "assets", "task_objects", "peg_scene.xml")
    urdf_path = os.path.join(project_root, "assets", "franka_panda", "robots", "panda_arm.urdf")

    print(f"正在加载 MuJoCo XML: {xml_path}")
    print(f"正在加载 URDF: {urdf_path}")

    # 2. 初始化双系统
    try:
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        mj_data = mujoco.MjData(mj_model)
    except Exception as e:
        print(f"MuJoCo 模型加载失败，请检查 XML 文件: {e}")
        return
    
    try:
        # URDF 中 Franka 机械臂的末端法兰通常命名为 panda_link8
        pin_model = PinocchioModel(urdf_path, ee_frame_name="panda_link8")
    except Exception as e:
        print(f"Pinocchio 模型加载失败，请检查 URDF 文件或 Frame 名称: {e}")
        return

    # 3. 赋予一个随机的非零关节构型 (7个关节)
    # 避免在全零的奇异点测试，更容易暴露坐标系定义偏差
    q_test = np.array([0.5, -0.3, 0.2, -1.5, 0.4, 1.2, 0.3])
    dq_test = np.zeros(7) 

    # 4. MuJoCo 系统计算
    mj_data.qpos[:7] = q_test
    mujoco.mj_kinematics(mj_model, mj_data) 
    
    # 尝试获取 MuJoCo 中的末端位置
    # DeepMind 的模型中连杆通常叫 "link8"，如果没有，尝试 "panda_link8"
    ee_id_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "link8")
    if ee_id_mj == -1: 
        ee_id_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "panda_link8")
    
    if ee_id_mj == -1:
        print("警告: 在 MuJoCo 模型中找不到 link8 或 panda_link8，请检查 xml 里的 body name")
        return
        
    ee_pos_mj = mj_data.xpos[ee_id_mj]

    # 5. Pinocchio 系统计算
    pin_model.update_state(q_test, dq_test)
    ee_pos_pin, _ = pin_model.get_ee_pose()

    # 6. 对比输出
    print("\n" + "="*50)
    print("【正运动学 (FK) 双盲测试结果】")
    print(f"MuJoCo    法兰位置 (xyz): {ee_pos_mj}")
    print(f"Pinocchio 法兰位置 (xyz): {ee_pos_pin}")
    
    error = np.linalg.norm(ee_pos_mj - ee_pos_pin)
    print(f"空间位置绝对误差: {error:.8f} 米")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()