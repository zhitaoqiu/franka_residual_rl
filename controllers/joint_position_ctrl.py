import numpy as np


class JointPositionController:
    def __init__(self, kp: float = 1.0, ctrl_limit: float = 0.5):
        self.kp = kp
        self.ctrl_limit = ctrl_limit

    def compute(self, q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        err = q_target - q_current
        ctrl = q_current + self.kp * err
        ctrl = np.clip(ctrl, q_current - self.ctrl_limit, q_current + self.ctrl_limit)
        return ctrl