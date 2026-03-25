from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dynamics_wrapper import backend_name as dynamics_backend_name
from dynamics_wrapper import step as dynamics_step
from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, STATE_DIM, VX, VY, VZ, WX, WY, WZ
from dynamics_wrapper.params import validate_params
from rl.adaptive_mpc_common import (
    SCHEDULER_OBS_DIM,
    build_scheduler_observation,
    build_scheduler_settings,
    compute_scheduler_reward,
    project_goal_with_lookahead,
    resolve_scheduler_action,
    sense_visible_obstacles,
    world_velocity,
)
from rl.configs.fish_env import build_fish_env_config
from rl.envs.geometry import nearest_obstacle_info, wrap_angle
from simulation.local_planning.fin_controller import extract_attitude, fin_controller
from simulation.local_planning.mpc_planner import MPCLocalPlanner


def _quat_from_forward_vector(forward_world: np.ndarray) -> np.ndarray:
    forward_world = np.asarray(forward_world, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(forward_world))
    if speed < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw = float(np.arctan2(forward_world[1], forward_world[0]))
    pitch = float(np.arctan2(forward_world[2], np.linalg.norm(forward_world[:2])))
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    quat = np.array([cp * cy, sp * sy, -sp * cy, cp * sy], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-8)


class MPCSchedulerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.cfg = deepcopy(config) if config is not None else build_fish_env_config()
        self.scheduler_cfg = build_scheduler_settings(self.cfg)
        self.dt = float(self.cfg["dt"])
        self.c_A = float(self.cfg["c_A"])
        self.fin_f = float(self.cfg["fin_f"])
        self.max_steps = int(self.scheduler_cfg["max_steps"])
        self.fish_radius = float(self.cfg["fish_radius"])
        self.obs_clip = float(self.cfg["obs_clip"])
        self.base_command_speed = float(self.scheduler_cfg["base_command_speed"])
        self.max_solver_failures = int(self.scheduler_cfg["max_consecutive_solver_fails"])
        self.propagation_model = str(self.scheduler_cfg.get("propagation_model", "kinematic")).lower()
        if self.propagation_model not in {"kinematic", "dynamic"}:
            raise ValueError(f"Unsupported scheduler propagation_model: {self.propagation_model}")
        kinematics_cfg = self.scheduler_cfg.get("kinematics", {})
        self.heading_tau = float(kinematics_cfg.get("heading_tau", self.cfg.get("mpc", {}).get("heading_tau", self.dt)))
        self.speed_tau = float(kinematics_cfg.get("speed_tau", self.cfg.get("mpc", {}).get("speed_tau", self.dt)))

        self.body_params = np.asarray(self.cfg["body_params"], dtype=np.float64)
        self.fin_params = np.asarray(self.cfg["fin_params"], dtype=np.float64)
        if self.propagation_model == "dynamic":
            validate_params(self.body_params, self.fin_params)
        self.initial_state_template = np.asarray(self.cfg["initial_state_template"], dtype=np.float64)
        if self.initial_state_template.shape != (STATE_DIM,):
            raise ValueError(
                f"initial_state_template must have shape ({STATE_DIM},), got {self.initial_state_template.shape}"
            )

        self.ref_min = np.asarray(self.cfg["ref_min"], dtype=np.float64)
        self.ref_max = np.asarray(self.cfg["ref_max"], dtype=np.float64)
        self.trim_refs = np.asarray(self.cfg["trim_refs"], dtype=np.float64)
        self.controller_params = deepcopy(self.cfg["controller_params"])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-self.obs_clip,
            high=self.obs_clip,
            shape=(SCHEDULER_OBS_DIM,),
            dtype=np.float32,
        )

        self.state = np.zeros(STATE_DIM, dtype=np.float64)
        self.hist = self.trim_refs.copy()
        self.goal_xyz = np.zeros(3, dtype=np.float64)
        self.obstacles: list[dict] = []
        self.scene_bounds_low = np.zeros(3, dtype=np.float64)
        self.scene_bounds_high = np.zeros(3, dtype=np.float64)
        self.prev_mode_name: str | None = None
        self.prev_obs_info: dict = {}
        self.prev_position = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.t_k = 0.0
        self.consecutive_solver_fails = 0
        self.backend = dynamics_backend_name() if self.propagation_model == "dynamic" else "kinematic"
        self.mpc_planner = MPCLocalPlanner(config=self.cfg)
        self.ctrl_state = self._build_ctrl_state()

    def _build_ctrl_state(self) -> dict:
        return {
            "e_theta_prev": 0.0,
            "e_theta_int": 0.0,
            "e_psi_prev": 0.0,
            "e_psi_int": 0.0,
        }

    def _step_kinematics(
        self,
        attitude_ref: np.ndarray,
        cmd_speed_ref: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        psi_ref = float(attitude_ref[0])
        theta_ref = float(attitude_ref[1])
        cmd_speed_ref = float(max(cmd_speed_ref, 0.0))

        current_state = np.asarray(self.state, dtype=np.float64).reshape(-1)
        current_position = current_state[[PX, PY, PZ]]
        psi, theta = extract_attitude(current_state)
        speed = float(max(current_state[VX], 0.0))

        alpha = min(1.0, self.dt / max(self.heading_tau, 1e-6))
        beta = min(1.0, self.dt / max(self.speed_tau, 1e-6))
        psi_next = psi + alpha * wrap_angle(psi_ref - psi)
        theta_next = theta + alpha * (theta_ref - theta)
        speed_next = speed + beta * (cmd_speed_ref - speed)

        forward_world = np.array(
            [
                np.cos(theta_next) * np.cos(psi_next),
                np.cos(theta_next) * np.sin(psi_next),
                -np.sin(theta_next),
            ],
            dtype=np.float64,
        )
        next_state = current_state.copy()
        next_state[[PX, PY, PZ]] = current_position + self.dt * speed_next * forward_world
        next_state[[VX, VY, VZ]] = np.array([speed_next, 0.0, 0.0], dtype=np.float64)
        next_state[[WX, WY, WZ]] = np.array(
            [0.0, (theta_next - theta) / self.dt, wrap_angle(psi_next - psi) / self.dt],
            dtype=np.float64,
        )
        next_state[Q0 : Q3 + 1] = _quat_from_forward_vector(forward_world)
        return next_state, self.hist.copy()

    def _sample_goal(self, start_xyz: np.ndarray) -> np.ndarray:
        low = float(self.scheduler_cfg["goal_distance_low"])
        high = float(self.scheduler_cfg["goal_distance_high"])
        distance = float(self.np_random.uniform(low, high))
        azimuth = float(self.np_random.uniform(-0.8, 0.8))
        elevation = float(self.np_random.uniform(-0.35, 0.35))
        direction = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ],
            dtype=np.float64,
        )
        return start_xyz + distance * direction

    def _sample_dynamic_velocity(self) -> np.ndarray:
        spawn = self.cfg["spawn"]
        speed = float(self.np_random.uniform(spawn["dynamic_speed_low"], spawn["dynamic_speed_high"]))
        direction = self.np_random.normal(size=3)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            direction = direction / direction_norm
        return speed * direction

    def _sample_obstacles(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> list[dict]:
        spawn = self.cfg["spawn"]
        num_obstacles = int(
            self.np_random.integers(spawn["num_obstacles_low"], spawn["num_obstacles_high"] + 1)
        )
        corridor = goal_xyz - start_xyz
        corridor_len = float(np.linalg.norm(corridor))
        corridor_dir = corridor / max(corridor_len, 1e-8)
        seed_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(corridor_dir, seed_axis))) > 0.95:
            seed_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        side = np.cross(corridor_dir, seed_axis)
        side = side / max(np.linalg.norm(side), 1e-8)
        lift = np.cross(corridor_dir, side)
        obstacles: list[dict] = []
        attempts = 0
        while len(obstacles) < num_obstacles and attempts < 200:
            attempts += 1
            radius = float(self.np_random.uniform(spawn["obstacle_radius_low"], spawn["obstacle_radius_high"]))
            along = float(self.np_random.uniform(0.15, 0.85)) * corridor_len
            lateral = float(self.np_random.uniform(-2.5, 2.5))
            vertical = float(self.np_random.uniform(-2.0, 2.0))
            center = start_xyz + corridor_dir * along + side * lateral + lift * vertical
            if np.linalg.norm(center - start_xyz) <= spawn["start_clearance"] + radius:
                continue
            if np.linalg.norm(center - goal_xyz) <= spawn["goal_clearance"] + radius:
                continue
            valid = True
            for obstacle in obstacles:
                min_gap = float(obstacle["r"]) + radius + spawn["obstacle_clearance"]
                if np.linalg.norm(center - obstacle["c"]) <= min_gap:
                    valid = False
                    break
            if not valid:
                continue
            obstacles.append(
                {
                    "c": center.astype(np.float64),
                    "r": radius,
                    "v": np.zeros(3, dtype=np.float64),
                    "track_id": len(obstacles),
                }
            )
        if obstacles:
            min_dynamic = min(int(spawn["dynamic_count_min"]), len(obstacles))
            dynamic_count = int(self.np_random.integers(min_dynamic, len(obstacles) + 1))
            dynamic_indices = self.np_random.choice(len(obstacles), size=dynamic_count, replace=False)
            for idx in np.asarray(dynamic_indices, dtype=np.int64):
                obstacles[int(idx)]["v"] = self._sample_dynamic_velocity()
        return obstacles

    def _reset_scene_bounds(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> None:
        margin = float(self.scheduler_cfg["scene_margin"])
        low = np.minimum(start_xyz, goal_xyz) - margin
        high = np.maximum(start_xyz, goal_xyz) + margin
        self.scene_bounds_low = low.astype(np.float64)
        self.scene_bounds_high = high.astype(np.float64)

    def _update_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle["c"] = np.asarray(obstacle["c"], dtype=np.float64) + np.asarray(obstacle["v"], dtype=np.float64) * self.dt
            legal_low = self.scene_bounds_low + float(obstacle["r"])
            legal_high = self.scene_bounds_high - float(obstacle["r"])
            for axis in range(3):
                if obstacle["c"][axis] < legal_low[axis]:
                    obstacle["c"][axis] = legal_low[axis]
                    obstacle["v"][axis] = abs(obstacle["v"][axis])
                elif obstacle["c"][axis] > legal_high[axis]:
                    obstacle["c"][axis] = legal_high[axis]
                    obstacle["v"][axis] = -abs(obstacle["v"][axis])

    def _get_visible_obstacles(self) -> tuple[list[dict], dict]:
        return sense_visible_obstacles(
            self.state[[PX, PY, PZ]],
            world_velocity(self.state),
            self.obstacles,
            self.scheduler_cfg,
        )

    def _build_obs(self, visible_obstacles: list[dict], prev_mode_for_obs: str | None) -> tuple[np.ndarray, dict]:
        return build_scheduler_observation(
            self.state,
            self.goal_xyz,
            visible_obstacles,
            self.fish_radius,
            prev_mode_for_obs,
            self.scheduler_cfg,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        self.step_count = 0
        self.t_k = 0.0
        self.consecutive_solver_fails = 0
        self.prev_mode_name = None
        self.hist = self.trim_refs.copy()
        self.ctrl_state = self._build_ctrl_state()
        self.mpc_planner.reset()

        start_xyz = self.np_random.uniform(
            self.cfg["spawn"]["start_xyz_low"],
            self.cfg["spawn"]["start_xyz_high"],
        ).astype(np.float64)
        self.goal_xyz = self._sample_goal(start_xyz)
        self.state = self.initial_state_template.copy()
        self.state[[PX, PY, PZ]] = start_xyz
        self.state[Q0 : Q3 + 1] = _quat_from_forward_vector(self.goal_xyz - start_xyz)
        self.obstacles = self._sample_obstacles(start_xyz, self.goal_xyz)
        self._reset_scene_bounds(start_xyz, self.goal_xyz)
        self.prev_position = start_xyz.copy()

        visible_obstacles, safe_zone = self._get_visible_obstacles()
        obs, obs_info = self._build_obs(visible_obstacles, prev_mode_for_obs=None)
        self.prev_obs_info = obs_info
        info = {
            "goal": self.goal_xyz.copy(),
            "goal_dist": obs_info["goal_dist"],
            "collision_risk": obs_info["collision_risk"],
            "target_mode": obs_info["target_mode"],
            "num_visible_obs": obs_info["num_visible_obs"],
            "safe_zone": safe_zone,
            "propagation_model": self.propagation_model,
            "backend": self.backend,
        }
        return obs, info

    def step(self, action):
        self.step_count += 1
        prev_mode = self.prev_mode_name
        selected_mode, strategy_params = resolve_scheduler_action(action, self.scheduler_cfg)
        visible_obstacles, safe_zone = self._get_visible_obstacles()
        prev_position = self.state[[PX, PY, PZ]].copy()
        local_target = project_goal_with_lookahead(prev_position, self.goal_xyz, float(strategy_params["lookahead"]))

        attitude_ref, cmd_speed_ref, planner_info = self.mpc_planner.plan(
            self.state,
            self.hist,
            local_target,
            visible_obstacles,
            self.base_command_speed,
            strategy_params=strategy_params,
        )
        if self.propagation_model == "dynamic":
            a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, self.ctrl_state = fin_controller(
                float(attitude_ref[0]),
                float(attitude_ref[1]),
                float(cmd_speed_ref),
                self.state,
                self.ctrl_state,
                self.dt,
                self.controller_params,
            )
            action_ref = np.clip(
                np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64),
                self.ref_min,
                self.ref_max,
            )
            next_state, next_hist = dynamics_step(
                self.state,
                self.body_params,
                self.fin_params,
                self.dt,
                self.t_k,
                action_ref,
                self.hist,
                self.c_A,
                self.fin_f,
            )
        else:
            self.ctrl_state = self._build_ctrl_state()
            action_ref = np.zeros_like(self.hist)
            next_state, next_hist = self._step_kinematics(attitude_ref, float(cmd_speed_ref))
        self.t_k += self.dt
        numerical_issue = not np.isfinite(next_state).all()
        if not numerical_issue:
            self.state = next_state
            self.hist = next_hist
            self._update_obstacles()

        current_position = self.state[[PX, PY, PZ]].copy()
        velocity_world = world_velocity(self.state)
        nearest_obstacle = nearest_obstacle_info(current_position, velocity_world, self.obstacles, self.fish_radius)
        clearance = float(nearest_obstacle["clearance"])
        reached_goal = float(np.linalg.norm(current_position - self.goal_xyz)) < float(self.cfg["goal_radius"])
        collided = clearance <= 0.0

        if planner_info.get("success", False):
            self.consecutive_solver_fails = 0
        else:
            self.consecutive_solver_fails += 1

        visible_next, safe_zone_next = self._get_visible_obstacles() if not numerical_issue else (visible_obstacles, safe_zone)
        obs, obs_info = self._build_obs(visible_next, prev_mode_for_obs=selected_mode)
        reward, reward_terms = compute_scheduler_reward(
            prev_position,
            current_position,
            self.goal_xyz,
            selected_mode,
            obs_info["target_mode"],
            clearance,
            bool(planner_info.get("success", False)) and not numerical_issue,
            prev_mode,
            reached_goal,
            collided or numerical_issue,
            self.scheduler_cfg,
        )
        if numerical_issue:
            reward -= 5.0

        terminated = bool(reached_goal or collided or numerical_issue or self.consecutive_solver_fails >= self.max_solver_failures)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        self.prev_mode_name = selected_mode
        self.prev_position = current_position.copy()
        self.prev_obs_info = obs_info

        info = {
            "selected_mode": selected_mode,
            "target_mode": obs_info["target_mode"],
            "strategy_params": strategy_params,
            "planner_success": bool(planner_info.get("success", False)) and not numerical_issue,
            "planner_info": planner_info,
            "local_target": local_target.copy(),
            "goal": self.goal_xyz.copy(),
            "goal_dist": obs_info["goal_dist"],
            "collision_risk": obs_info["collision_risk"],
            "ttc": float(obs_info["dangerous_obstacle"]["ttc"]),
            "closing_speed": float(obs_info["dangerous_obstacle"]["closing_speed"]),
            "is_static": bool(obs_info["dangerous_obstacle"]["is_static"]),
            "moving_away": bool(obs_info["dangerous_obstacle"]["moving_away"]),
            "clearance": clearance,
            "num_visible_obs": obs_info["num_visible_obs"],
            "reward_terms": reward_terms,
            "reached_goal": reached_goal,
            "collided": collided,
            "numerical_issue": numerical_issue,
            "safe_zone": safe_zone_next,
            "cmd_speed_ref": float(cmd_speed_ref),
            "propagation_model": self.propagation_model,
            "action_ref": action_ref.copy(),
            "backend": self.backend,
        }
        return obs, float(reward), terminated, truncated, info
