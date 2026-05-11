# filepath: controllers/admittance_ctrl.py
import numpy as np

class AdmittanceController:
    def __init__(self, mass, damping, stiffness, dt):
        """
        笛卡尔空间导纳控制器 (Mass-Damper-Spring 模型)
        公式: M*ddx + D*dx + K*(x - x_target) = F_ext
        
        :param mass: 虚拟质量矩阵 (6x6) 或 标量
        :param damping: 虚拟阻尼矩阵 (6x6) 或 标量
        :param stiffness: 虚拟刚度矩阵 (6x6) 或 标量
        :param dt: 控制周期
        """
        self.dt = dt
        
        # 扩展为 6 维对角矩阵 (3位置 + 3姿态)
        self.M = mass * np.eye(6) if isinstance(mass, (int, float)) else mass
        self.D = damping * np.eye(6) if isinstance(damping, (int, float)) else damping
        self.K = stiffness * np.eye(6) if isinstance(stiffness, (int, float)) else stiffness
        
        self.M_inv = np.linalg.inv(self.M)
        
        # 内部状态：柔顺位姿和柔顺速度
        self.x_c = np.zeros(6)  
        self.dx_c = np.zeros(6)
        
        self.is_initialized = False

    def step(self, x_target, current_pose, f_ext):
        """
        计算单步的柔顺目标位姿
        :param x_target: 原始期望位姿 (6,) [x,y,z, rx,ry,rz]
        :param current_pose: 当前实际位姿 (6,) 
        :param f_ext: 末端受到的外部接触力/力矩 (6,)
        :return: 修正后的柔顺目标位姿 (6,)
        """
        if not self.is_initialized:
            self.x_c = current_pose.copy()
            self.is_initialized = True

        # 弹簧力：试图将末端拉向原始目标点
        f_spring = self.K @ (x_target - self.x_c)
        
        # 阻尼力：抑制运动过快
        f_damper = self.D @ (0 - self.dx_c) 

        # 导纳核心方程：计算柔顺加速度
        # ddx_c = M^-1 * (F_ext + F_spring + F_damper)
        ddx_c = self.M_inv @ (f_ext + f_spring + f_damper)

        # 欧拉积分更新速度和位置
        self.dx_c += ddx_c * self.dt
        self.x_c += self.dx_c * self.dt

        return self.x_c.copy()