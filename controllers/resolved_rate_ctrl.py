# filepath: controllers/resolved_rate_ctrl.py
import numpy as np
import pinocchio as pin


class ResolvedRateController:
    def __init__(self, pin_model, damping=0.03, kp_pos=2.0, kp_ori=10.0):
        """
        基于 DLS 的分解速率伺服控制器
        kp_ori 提高到 10.0，强力约束末端姿态，防止手腕在运动中倾斜
        """
        self.pin_model = pin_model
        self.damping = damping
        self.kp_pos = kp_pos
        self.kp_ori = kp_ori

    def compute_joint_velocity(self, q, dq, target_pos, target_rot):
        self.pin_model.update_state(q, dq)
        curr_pos, curr_rot = self.pin_model.get_ee_pose()

        pos_err = target_pos - curr_pos

        R_err = curr_rot.T @ target_rot
        ori_err_local = pin.log3(R_err)
        ori_err_world = curr_rot @ ori_err_local

        v_d = np.zeros(6)
        v_d[:3] = self.kp_pos * pos_err
        v_d[3:] = self.kp_ori * ori_err_world

        J = self.pin_model.get_ee_jacobian()

        lambda_sq = self.damping ** 2
        JJt = J @ J.T
        dq_out = J.T @ np.linalg.inv(JJt + lambda_sq * np.eye(6)) @ v_d

        return dq_out, pos_err