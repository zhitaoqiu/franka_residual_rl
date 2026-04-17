import numpy as np
import pinocchio as pin


class PinocchioModel:
    def __init__(self, urdf_path: str, ee_frame_name: str = "panda_link8"):
        self.urdf_path = urdf_path
        self.ee_frame_name = ee_frame_name

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)

    def forward_kinematics(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

    def get_ee_position(self, q: np.ndarray) -> np.ndarray:
        self.forward_kinematics(q)
        return self.data.oMf[self.ee_frame_id].translation.copy()

    def get_frame_jacobian(self, q: np.ndarray) -> np.ndarray:
        self.forward_kinematics(q)
        J = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return J.copy()