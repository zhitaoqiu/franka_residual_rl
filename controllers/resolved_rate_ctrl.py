import numpy as np


class ResolvedRateController:
    def __init__(self, kp: float = 1.0, dq_limit: float = 0.05):
        self.kp = kp
        self.dq_limit = dq_limit

    def compute_q_target(
        self,
        q_current: np.ndarray,
        ee_pos: np.ndarray,
        target_pos: np.ndarray,
        J_full: np.ndarray,
    ) -> np.ndarray:
        q_current = np.asarray(q_current, dtype=np.float64).reshape(-1)
        ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)

        # 只取位置雅可比
        J_pos = J_full[:3, :]   # shape: (3, 7)

        pos_err = target_pos - ee_pos
        dx = self.kp * pos_err

        # 伪逆求解 dq
        dq = np.linalg.pinv(J_pos) @ dx

        # 限幅，避免一步太大
        dq = np.clip(dq, -self.dq_limit, self.dq_limit)

        q_target = q_current + dq
        return q_target