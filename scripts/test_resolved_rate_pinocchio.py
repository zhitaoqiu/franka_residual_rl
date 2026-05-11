# filepath: scripts/test_resolved_rate_pinocchio.py
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import pinocchio as pin

# 自动处理路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from controllers.pinocchio_model import PinocchioModel
from controllers.resolved_rate_ctrl import ResolvedRateController

def main():
    # 路径匹配你的项目结构
    xml_path = os.path.join(project_root, "assets", "franka_panda", "franka_emika_panda", "scene.xml")
    urdf_path = os.path.join(project_root, "assets", "franka_panda", "robots", "panda_arm.urdf")
    
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    pin_model = PinocchioModel(urdf_path, ee_frame_name="panda_link8")
    
    # 初始化控制器：设置合适的增益
    controller = ResolvedRateController(pin_model, damping=0.02, kp_pos=3.0, kp_ori=3.0)

    # 设定目标：插孔上方 15cm，末端垂直向下
    target_pos = np.array([0.5, 0.0, 0.35]) 
    target_rot = pin.utils.rpyToMatrix(np.pi, 0, 0) # 绕 Y 轴转 180 度使其朝下

    # 初始化机器人姿态 (常见的避开奇异点的初始姿态)
    mj_data.qpos[:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    mujoco.mj_forward(mj_model, mj_data)

    print(f">>> 启动实时伺服测试。目标位置: {target_pos}")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            loop_start = time.time()

            # 1. 获取当前状态
            q = mj_data.qpos[:7].copy()
            dq = mj_data.qvel[:7].copy()

            # 2. 计算期望关节速度
            joint_v_d, err = controller.compute_joint_velocity(q, dq, target_pos, target_rot)

            # 3. 运动学积分步进 (Kinematic Step)
            dt = mj_model.opt.timestep
            mj_data.qpos[:7] += joint_v_d * dt
            mj_data.qvel[:7] = joint_v_d

            # 4. 更新物理引擎并渲染
            mujoco.mj_kinematics(mj_model, mj_data)
            viewer.sync()

            # 打印收敛情况
            dist = np.linalg.norm(err)
            if dist < 0.001:
                print("目标已精准送达！")
                break

            # 严格控制循环频率
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

if __name__ == "__main__":
    main()