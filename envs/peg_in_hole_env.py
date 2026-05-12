# filepath: envs/peg_in_hole_env.py
"""
Simplified z-insertion RL task (Version 1).

Design goals for this stage:
1) Skip Pinocchio during RL contact phase.
2) Reset peg directly above hole center with vertical-down orientation.
3) RL only controls z insertion.
4) Observation/reward/termination are simple and debug-friendly.

Future extension path:
- V2: add tiny x/y residual.
- V3: random x/y reset bias.
- V4: add tiny orientation bias.
- V5+: re-connect Pinocchio only for pre-insertion planning.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PegInHoleResidualEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        xml_path = os.path.join(
            project_root,
            "assets",
            "franka_panda",
            "franka_emika_panda",
            "scene_peg_in_hole.xml",
        )

        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # ----- ids -----
        self.peg_tip_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip"
        )
        self.hole_center_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "hole_center"
        )
        self.ft_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "ft_sensor_site"
        )
        self.peg_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "peg_geom"
        )
        self.wall_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "wall_0"
        )
        # Body IDs for dynamics randomization
        self._arm_body_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
            for i in range(8)
        ]
        self._hand_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        # Store nominal dynamics for reset
        self._nominal_dof_damping = self.mj_model.dof_damping.copy()
        self._nominal_geom_friction = self.mj_model.geom_friction.copy()
        self._nominal_body_mass = self.mj_model.body_mass.copy()

        # ----- action / observation -----
        # action = [v_z_cmd], positive value means move down.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # obs = [z_err, v_z, Fz, Fx, Fy, last_action]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        # ----- task parameters -----
        self.render_mode = render_mode
        self.max_steps = 600
        self.control_decimation = 5
        self.q_noise = 0.005

        # ----- domain randomization -----
        self.enable_obs_noise = False       # force noise hurts more than it helps for this task
        self.obs_force_noise_std = 0.3       # N, Gaussian noise on fx/fy/fz
        self.enable_dynamics_randomization = True
        self.dyn_mass_range = (0.95, 1.05)   # body mass multiplier
        self.dyn_damping_range = (0.9, 1.1)  # joint damping multiplier
        self.dyn_friction_range = (0.7, 1.5) # geom friction multiplier
        self.enable_init_xy_randomization = True
        self.init_xy_range = 0.002           # ±2mm XY offset at reset

        self.safe_offset = 0.05
        self.target_insertion_depth = 0.025
        self.max_vz = 0.06
        self.max_retract = 0.02
        self.max_overshoot = 0.005

        self.soft_fz_limit = 55.0
        self.hard_fz_limit = 120.0
        self.soft_lateral_force_limit = 35.0
        self.hard_lateral_force_limit = 70.0

        self.progress_weight = 80.0
        self.time_penalty = 0.01
        self.force_penalty_weight = 0.002   # relaxed — insertion needs some force
        self.lateral_penalty_weight = 0.005
        self.action_penalty_weight = 0.02
        self.action_smooth_penalty_weight = 0.01
        self.success_bonus = 25.0
        self.failure_penalty = 20.0

        self.success_z_tol = 0.002  # tight tolerance for precision insertion
        self.success_force_limit = 35.0
        self.success_lateral_limit = 20.0

        # Vertical-down target axis for peg tip local z-axis.
        self.target_tool_z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Nominal aligned q from local IK solve; used as reset seed.
        self.aligned_q_nominal = np.array(
            [0.0, -0.04411257, 0.0, -2.332463, 0.0, 2.288347, 0.785],
            dtype=np.float64,
        )

        # Keep gripper commanded open so finger contacts do not dominate FT signal.
        self.gripper_hold_ctrl = 255.0

        # ----- runtime state -----
        self.step_count = 0
        self.last_action = 0.0
        self.prev_depth_error = 0.0
        self.termination_reason = "running"
        self.pinocchio_enabled_for_pre_insertion = False
        self.ft_bias_world = np.zeros(6, dtype=np.float64)
        self.episode_log: List[Dict[str, float]] = []

        self.target_xy = np.zeros(2, dtype=np.float64)
        self.target_z = 0.0
        self.goal_z = 0.0
        self.start_z = 0.0
        self.hole_top_z = 0.0
        self.desired_tip_rot = np.eye(3, dtype=np.float64)

        # Persistent joint target for MuJoCo position actuators.
        # Important: do not rebuild q_target from current q every step.
        self.joint_target_q = self.aligned_q_nominal.copy()

    # ------------------------------------------------------------------
    # Placeholder hook for future integration (not used in V1).
    # ------------------------------------------------------------------
    def set_pinocchio_pre_insertion_enabled(self, enabled: bool):
        self.pinocchio_enabled_for_pre_insertion = bool(enabled)

    # ------------------------------------------------------------------
    # Geometry / state helpers
    # ------------------------------------------------------------------
    def _get_hole_center_world(self) -> np.ndarray:
        return self.mj_data.site_xpos[self.hole_center_site_id].copy()

    def _get_hole_top_z(self) -> float:
        # top = world z center + half height of wall geom
        return float(
            self.mj_data.geom_xpos[self.wall_geom_id, 2]
            + self.mj_model.geom_size[self.wall_geom_id, 2]
        )

    def _get_peg_tip_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.mj_data.site_xpos[self.peg_tip_site_id].copy()
        rot = self.mj_data.site_xmat[self.peg_tip_site_id].copy().reshape(3, 3)
        return pos, rot

    def _get_peg_tip_vz(self) -> float:
        jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.peg_tip_site_id)
        return float(jacp[2, :7] @ self.mj_data.qvel[:7])

    def _get_wrench_world(self) -> np.ndarray:
        # MuJoCo force/torque sensors are in site local frame.
        # Convert to world frame and use sign convention: environment-on-tool.
        force_local = self.mj_data.sensor("ee_force_sensor").data.copy()
        torque_local = self.mj_data.sensor("ee_torque_sensor").data.copy()
        rot_world_from_local = self.mj_data.site_xmat[self.ft_site_id].copy().reshape(3, 3)

        wrench_world_raw = np.concatenate(
            [rot_world_from_local @ force_local, rot_world_from_local @ torque_local]
        )
        # FT sensor at wrist already includes peg contact forces transmitted through
        # the rigid hand chain. Do NOT add direct contact forces — that double-counts.
        return -(wrench_world_raw - self.ft_bias_world)

    def _get_sensor_only_wrench_world(self) -> np.ndarray:
        force_local = self.mj_data.sensor("ee_force_sensor").data.copy()
        torque_local = self.mj_data.sensor("ee_torque_sensor").data.copy()
        rot_world_from_local = self.mj_data.site_xmat[self.ft_site_id].copy().reshape(3, 3)

        wrench_world_raw = np.concatenate(
            [rot_world_from_local @ force_local, rot_world_from_local @ torque_local]
        )
        return -(wrench_world_raw - self.ft_bias_world)

    def _get_peg_contact_force_world(self) -> np.ndarray:
        total_force = np.zeros(3, dtype=np.float64)
        cf = np.zeros(6, dtype=np.float64)

        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]

            if c.geom1 != self.peg_geom_id and c.geom2 != self.peg_geom_id:
                continue

            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, cf)
            frame = c.frame.reshape(3, 3)
            f_world = frame @ cf[:3]

            # Contact normal points from geom1 to geom2.
            # Convert to force on peg in world frame.
            if c.geom1 == self.peg_geom_id:
                total_force += -f_world
            else:
                total_force += f_world

        return total_force

    @staticmethod
    def _rotation_error_world(curr_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        # SO(3) error represented in world frame.
        # e = 0.5 * sum_i (R_c[:, i] x R_t[:, i])
        return 0.5 * (
            np.cross(curr_rot[:, 0], target_rot[:, 0])
            + np.cross(curr_rot[:, 1], target_rot[:, 1])
            + np.cross(curr_rot[:, 2], target_rot[:, 2])
        )

    @staticmethod
    def _make_vertical_down_rot(curr_rot: np.ndarray) -> np.ndarray:
        """
        Construct a valid target rotation whose local z-axis points straight down
        in world frame. Keep the current yaw as much as possible by projecting
        the current local x-axis onto the horizontal plane.
        """
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        x_hint = curr_rot[:, 0].copy()
        x_axis = x_hint - z_axis * np.dot(x_hint, z_axis)

        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)

        # Columns are local axes in world frame.
        # Need right-handed frame: x cross y = z.
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

        return np.column_stack([x_axis, y_axis, z_axis])

    def _compute_depth_error(self, tip_z: float) -> float:
        # Remaining insertion error; 0 means reached/exceeded goal depth.
        return max(0.0, tip_z - self.goal_z)

    # ------------------------------------------------------------------
    # Requested module 1: reset_to_aligned_pose()
    # ------------------------------------------------------------------
    def reset_to_aligned_pose(self, xy_offset: np.ndarray | None = None):
        hole_center = self._get_hole_center_world()
        self.hole_top_z = self._get_hole_top_z()

        self.target_xy = hole_center[:2].copy()

        if xy_offset is not None:
            self.target_xy = self.target_xy + xy_offset

        desired_z = self.hole_top_z + self.safe_offset
        desired_pos = np.array(
            [self.target_xy[0], self.target_xy[1], desired_z],
            dtype=np.float64,
        )

        q = self.mj_data.qpos[:7].copy()
        target_rot = self.desired_tip_rot

        for _ in range(120):
            self.mj_data.qpos[:7] = q
            self.mj_data.qvel[:7] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

            tip_pos, tip_rot = self._get_peg_tip_pose()
            pos_err = desired_pos - tip_pos
            ori_err = self._rotation_error_world(tip_rot, target_rot)

            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(ori_err) < 1e-3:
                break

            jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.peg_tip_site_id)

            # Weighted task: xyz + full orientation lock.
            ori_w = 1.0
            task = np.concatenate([pos_err, ori_w * ori_err])
            J = np.vstack([jacp[:, :7], ori_w * jacr[:, :7]])

            damp = 1e-2
            dq = J.T @ np.linalg.solve(J @ J.T + (damp**2) * np.eye(6), task)

            q = q + 0.6 * dq
            q = np.clip(
                q,
                self.mj_model.actuator_ctrlrange[:7, 0],
                self.mj_model.actuator_ctrlrange[:7, 1],
            )

        self.mj_data.qpos[:7] = q
        self.mj_data.qvel[:7] = 0.0

        # Important: for MuJoCo position actuators, ctrl is a persistent joint target.
        # Do not let the target follow the current q every step.
        self.joint_target_q = q.copy()
        self.mj_data.ctrl[:7] = self.joint_target_q

        if self.mj_model.nu > 7:
            self.mj_data.ctrl[7] = self.gripper_hold_ctrl

        mujoco.mj_forward(self.mj_model, self.mj_data)

        tip_pos, _ = self._get_peg_tip_pose()
        self.start_z = float(tip_pos[2])
        self.target_z = self.start_z
        self.goal_z = self.hole_top_z - self.target_insertion_depth

    # ------------------------------------------------------------------
    # Requested module 2: get_observation()
    # ------------------------------------------------------------------
    def get_observation(self) -> np.ndarray:
        tip_pos, _ = self._get_peg_tip_pose()
        vz = self._get_peg_tip_vz()
        wrench = self._get_wrench_world()

        z_error = tip_pos[2] - self.goal_z
        fx, fy, fz = float(wrench[0]), float(wrench[1]), float(wrench[2])

        # Observation noise on force channels (sim-to-real)
        if self.enable_obs_noise:
            fx += self.np_random.normal(0.0, self.obs_force_noise_std)
            fy += self.np_random.normal(0.0, self.obs_force_noise_std)
            fz += self.np_random.normal(0.0, self.obs_force_noise_std)

        # Fixed-scale normalization so all dims are roughly [-1, 1].
        obs = np.array(
            [
                z_error / 0.1,
                vz / 0.5,
                fz / 45.0,
                fx / 45.0,
                fy / 45.0,
                self.last_action,
            ],
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Requested module 3: apply_action(action)
    # ------------------------------------------------------------------
    def apply_action(self, action: np.ndarray):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0).item())
        dt = self.mj_model.opt.timestep * self.control_decimation

        # Positive action -> downward command, meaning smaller z in world frame.
        self.target_z = self.target_z - a * self.max_vz * dt
        self.target_z = float(
            np.clip(
                self.target_z,
                self.goal_z - self.max_overshoot,
                self.start_z + self.max_retract,
            )
        )

        for _ in range(self.control_decimation):
            tip_pos, tip_rot = self._get_peg_tip_pose()

            err_xy = self.target_xy - tip_pos[:2]
            err_z = self.target_z - tip_pos[2]
            ori_err = self._rotation_error_world(tip_rot, self.desired_tip_rot)
            vz = self._get_peg_tip_vz()

            v_task = np.array(
                [
                    10.0 * err_xy[0],
                    10.0 * err_xy[1],
                    14.0 * err_z - 0.8 * vz,
                ],
                dtype=np.float64,
            )
            w_task = 4.5 * ori_err

            jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.peg_tip_site_id)

            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            task = np.concatenate([v_task, w_task])

            damp = 2e-2
            dq_cmd = J.T @ np.linalg.solve(J @ J.T + (damp**2) * np.eye(6), task)
            dq_cmd = np.clip(dq_cmd, -0.8, 0.8)

            # Integrate the IK velocity into a persistent joint target.
            # This is crucial for MuJoCo position actuators.
            self.joint_target_q = self.joint_target_q + dq_cmd * self.mj_model.opt.timestep

            self.joint_target_q = np.clip(
                self.joint_target_q,
                self.mj_model.actuator_ctrlrange[:7, 0],
                self.mj_model.actuator_ctrlrange[:7, 1],
            )

            ctrl = np.zeros(self.mj_model.nu, dtype=np.float64)
            ctrl[:7] = self.joint_target_q

            if self.mj_model.nu > 7:
                ctrl[7] = self.gripper_hold_ctrl

            self.mj_data.ctrl[:] = ctrl
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.last_action = a

    # ------------------------------------------------------------------
    # Requested module 4: compute_reward()
    # ------------------------------------------------------------------
    def compute_reward(self) -> float:
        tip_pos, _ = self._get_peg_tip_pose()
        wrench = self._get_wrench_world()

        fx, fy, fz = float(wrench[0]), float(wrench[1]), float(wrench[2])

        curr_depth_error = self._compute_depth_error(float(tip_pos[2]))
        progress = self.prev_depth_error - curr_depth_error
        self.prev_depth_error = curr_depth_error

        lateral_force = float(np.linalg.norm([fx, fy]))

        force_penalty = self.force_penalty_weight * max(
            0.0,
            abs(fz) - self.soft_fz_limit,
        )
        lateral_penalty = self.lateral_penalty_weight * max(
            0.0,
            lateral_force - self.soft_lateral_force_limit,
        )
        action_penalty = self.action_penalty_weight * (self.last_action**2)

        last_last_action = self.episode_log[-1]["action"] if self.episode_log else 0.0
        smooth_penalty = self.action_smooth_penalty_weight * (
            (self.last_action - last_last_action) ** 2
        )

        reward = (
            self.progress_weight * progress
            - self.time_penalty
            - force_penalty
            - lateral_penalty
            - action_penalty
            - smooth_penalty
        )

        return float(reward)

    # ------------------------------------------------------------------
    # Requested module 5: check_termination()
    # ------------------------------------------------------------------
    def check_termination(self) -> Tuple[bool, bool]:
        tip_pos, _ = self._get_peg_tip_pose()
        wrench = self._get_wrench_world()

        fx, fy, fz = float(wrench[0]), float(wrench[1]), float(wrench[2])

        lateral_force = float(np.linalg.norm([fx, fy]))
        depth_error = self._compute_depth_error(float(tip_pos[2]))

        # V1 debug success:
        # First verify whether the peg can geometrically reach the target depth.
        # Do not use strict force thresholds yet.
        success = (
            depth_error <= self.success_z_tol
            and abs(fz) <= self.hard_fz_limit
            and lateral_force <= self.hard_lateral_force_limit
        )
        if success:
            self.termination_reason = "success"
            return True, False

        if abs(fz) > self.hard_fz_limit:
            self.termination_reason = "hard_fz_limit"
            return True, False

        if lateral_force > self.hard_lateral_force_limit:
            self.termination_reason = "hard_lateral_force_limit"
            return True, False

        if tip_pos[2] > self.start_z + self.max_retract:
            self.termination_reason = "peg_retracted_too_much"
            return True, False

        if self.step_count >= self.max_steps:
            self.termination_reason = "timeout"
            return False, True

        self.termination_reason = "running"
        return False, False

    # ------------------------------------------------------------------
    # Requested module 6: debug_logging()
    # ------------------------------------------------------------------
    def debug_logging(self, reward: float):
        tip_pos, _ = self._get_peg_tip_pose()
        wrench = self._get_wrench_world()

        fx, fy, fz = float(wrench[0]), float(wrench[1]), float(wrench[2])

        self.episode_log.append(
            {
                "step": float(self.step_count),
                "z_tip": float(tip_pos[2]),
                "z_goal": float(self.goal_z),
                "fx": fx,
                "fy": fy,
                "fz": fz,
                "reward": float(reward),
                "action": float(self.last_action),
            }
        )

    # ------------------------------------------------------------------
    # Domain randomization helpers
    # ------------------------------------------------------------------
    def disable_randomization(self):
        """Disable all domain randomization (for deterministic evaluation)."""
        self.enable_obs_noise = False
        self.enable_dynamics_randomization = False
        self.enable_init_xy_randomization = False

    def _randomize_dynamics(self):
        """Randomize joint damping, body masses, and geom friction each episode."""
        if not self.enable_dynamics_randomization:
            return

        rng = self.np_random

        # Joint damping (DOFs 0-6 are arm joints)
        for i in range(7):
            factor = rng.uniform(*self.dyn_damping_range)
            self.mj_model.dof_damping[i] = self._nominal_dof_damping[i] * factor

        # Body mass (arm links + hand)
        for bid in self._arm_body_ids + [self._hand_body_id]:
            factor = rng.uniform(*self.dyn_mass_range)
            self.mj_model.body_mass[bid] = self._nominal_body_mass[bid] * factor

        # Geom friction (peg + all wall geoms)
        factor = rng.uniform(*self.dyn_friction_range)
        self.mj_model.geom_friction[self.peg_geom_id] = (
            self._nominal_geom_friction[self.peg_geom_id] * factor
        )

        for i in range(36):
            try:
                wid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{i}")
            except Exception:
                continue
            factor = rng.uniform(*self.dyn_friction_range)
            self.mj_model.geom_friction[wid] = self._nominal_geom_friction[wid] * factor

    def _randomize_initial_xy(self) -> np.ndarray:
        """Return random XY offset for peg initial position."""
        if not self.enable_init_xy_randomization:
            return np.zeros(2, dtype=np.float64)
        return self.np_random.uniform(
            -self.init_xy_range, self.init_xy_range, size=2
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Restore nominal dynamics before re-randomizing
        self.mj_model.dof_damping[:] = self._nominal_dof_damping
        self.mj_model.body_mass[:] = self._nominal_body_mass
        self.mj_model.geom_friction[:] = self._nominal_geom_friction
        self._randomize_dynamics()
        self._xy_offset = self._randomize_initial_xy()

        self.step_count = 0
        self.last_action = 0.0
        self.termination_reason = "running"
        self.episode_log = []
        self.ft_bias_world = np.zeros(6, dtype=np.float64)

        q0 = self.aligned_q_nominal + self.np_random.uniform(
            -self.q_noise,
            self.q_noise,
            size=7,
        )

        self.mj_data.qpos[:7] = q0
        self.mj_data.qvel[:7] = 0.0

        if self.mj_model.nq > 8:
            self.mj_data.qpos[7] = 0.04
            self.mj_data.qpos[8] = 0.04

        self.joint_target_q = q0.copy()
        self.mj_data.ctrl[:7] = self.joint_target_q

        if self.mj_model.nu > 7:
            self.mj_data.ctrl[7] = self.gripper_hold_ctrl

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Force peg tip local z-axis to point straight down in world frame.
        _, tip_rot0 = self._get_peg_tip_pose()
        self.desired_tip_rot = self._make_vertical_down_rot(tip_rot0)

        # Align above hole center (with optional XY randomization) using Jacobian IK.
        self.reset_to_aligned_pose(xy_offset=self._xy_offset)

        # Ensure reset servo target and actual q are synchronized.
        self.joint_target_q = self.mj_data.qpos[:7].copy()
        self.mj_data.ctrl[:7] = self.joint_target_q

        if self.mj_model.nu > 7:
            self.mj_data.ctrl[7] = self.gripper_hold_ctrl

        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Bias FT sensor at free-space aligned pose.
        force_local = self.mj_data.sensor("ee_force_sensor").data.copy()
        torque_local = self.mj_data.sensor("ee_torque_sensor").data.copy()
        rot_world_from_local = self.mj_data.site_xmat[self.ft_site_id].copy().reshape(3, 3)

        self.ft_bias_world = np.concatenate(
            [rot_world_from_local @ force_local, rot_world_from_local @ torque_local]
        )

        tip_pos, _ = self._get_peg_tip_pose()
        self.prev_depth_error = self._compute_depth_error(float(tip_pos[2]))

        obs = self.get_observation()
        info = {
            "task_mode": "simplified_z_insertion_v1",
            "pinocchio_used_in_rl": False,
            "target_xy": self.target_xy.copy(),
            "start_z": float(self.start_z),
            "goal_z": float(self.goal_z),
            "hole_top_z": float(self.hole_top_z),
            "termination_reason": self.termination_reason,
            "xy_offset": self._xy_offset.copy(),
            "obs_noise": self.enable_obs_noise,
            "dyn_randomized": self.enable_dynamics_randomization,
        }

        return obs, info

    def step(self, action):
        self.step_count += 1
        self.apply_action(action)

        reward = self.compute_reward()
        terminated, truncated = self.check_termination()

        # Terminal shaping
        if self.termination_reason == "success":
            reward += self.success_bonus
        elif self.termination_reason not in ("running", "timeout"):
            reward -= self.failure_penalty

        self.debug_logging(reward)
        obs = self.get_observation()

        tip_pos, _ = self._get_peg_tip_pose()
        wrench = self._get_wrench_world()
        sensor_wrench = self._get_sensor_only_wrench_world()
        contact_force_world = self._get_peg_contact_force_world()

        fx, fy, fz = float(wrench[0]), float(wrench[1]), float(wrench[2])
        lateral_force = float(np.linalg.norm([fx, fy]))
        depth_error = self._compute_depth_error(float(tip_pos[2]))

        info = {
            "step_count": self.step_count,
            "termination_reason": self.termination_reason,
            "z_tip": float(tip_pos[2]),
            "z_goal": float(self.goal_z),
            "depth_error": float(depth_error),
            "contact_force": float(np.linalg.norm(wrench[:3])),
            "distance_to_target": float(depth_error),  # backward-compatible key
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "fz_sensor": float(sensor_wrench[2]),
            "fz_contact": float(contact_force_world[2]),
            "lateral_force": lateral_force,
            "target_z": float(self.target_z),
            "last_action": float(self.last_action),
            "debug_log_len": len(self.episode_log),
        }

        return obs, float(reward), terminated, truncated, info


# Optional alias for clearer external naming.
SimplifiedZInsertionEnv = PegInHoleResidualEnv