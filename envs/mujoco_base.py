'''把 MuJoCo 最基础、所有任务都会重复用到的东西，统一封装起来'''
import os
import numpy as np
import mujoco


class MujocoBase:
    def __init__(self, model_path: str, ee_body_name: str = "panda_hand"):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        self.ee_body_name = ee_body_name
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name
        )

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

    def reset(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def step_sim(self, ctrl, n_substeps: int = 1):
        ctrl = np.asarray(ctrl, dtype=np.float64)
        assert ctrl.shape[0] == self.nu, f"ctrl dim {ctrl.shape[0]} != nu {self.nu}"

        self.data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def get_joint_state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def get_ee_pose(self):
        xpos = self.data.xpos[self.ee_body_id].copy()
        xmat = self.data.xmat[self.ee_body_id].copy().reshape(3, 3)
        return xpos, xmat

    def get_obs(self):
        qpos, qvel = self.get_joint_state()
        ee_pos, ee_rot = self.get_ee_pose()
        return {
            "qpos": qpos,
            "qvel": qvel,
            "ee_pos": ee_pos,
            "ee_rot": ee_rot,
        }