# filepath: controllers/pinocchio_model.py
import pinocchio as pin
import numpy as np
import os

class PinocchioModel:
    def __init__(self, urdf_path: str, ee_frame_name: str = "panda_link8"):
        """
        初始化 Pinocchio 动力学模型
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

        # 加载模型与数据缓存区
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # 获取末端执行器的 Frame ID
        if self.model.existFrame(ee_frame_name):
            self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        else:
            raise ValueError(f"URDF 中不存在 frame: {ee_frame_name}")

    def update_state(self, q: np.ndarray, dq: np.ndarray):
        """
        更新机器人的关节位置与速度，预计算所有运动学/动力学变量
        """
        # 这一步极其关键，它会一次性算好所有的 Jacobian, FK, Gravity 等
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

    def get_ee_pose(self):
        """
        获取末端执行器的正运动学 (FK) 位姿
        :return: position (3,), rotation_matrix (3, 3)
        """
        ee_placement = self.data.oMf[self.ee_frame_id]
        return ee_placement.translation.copy(), ee_placement.rotation.copy()

    def get_ee_jacobian(self) -> np.ndarray:
        """
        获取末端执行器在世界坐标系下的雅可比矩阵 (6x7)
        """
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return J.copy()

    def get_gravity_compensation(self) -> np.ndarray:
        """获取重力补偿项"""
        return self.data.g.copy()