from __future__ import annotations

"""CasADi-based local MPC planner."""

import casadi as ca
import numpy as np

from dynamics.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, VX, VY, VZ
from rl_training.config.fish_env_config import build_fish_env_config
from rl_training.envs.geometry import quaternion_to_rotation_matrix, rotate_body_to_world
from simulation.fin_controller.controller import extract_attitude


POS_DIM = 3
CONTROL_DIM = 3


def _smooth_positive_part(value: ca.MX) -> ca.MX:
    return 0.5 * (value + ca.sqrt(value * value + 1e-6))


def _wrap_angle_mx(angle: ca.MX) -> ca.MX:
    return ca.atan2(ca.sin(angle), ca.cos(angle))


def _clip_mx(value: ca.MX, lower: float, upper: float) -> ca.MX:
    return ca.fmin(ca.fmax(value, lower), upper)


def _command_velocity(psi: ca.MX, theta: ca.MX, speed: ca.MX) -> ca.MX:
    return speed * ca.vertcat(
        ca.cos(theta) * ca.cos(psi),
        ca.cos(theta) * ca.sin(psi),
        -ca.sin(theta),
    )


class MPCLocalPlanner:
    def __init__(self, config: dict | None = None):
        self.config = build_fish_env_config() if config is None else config
        self.dt = float(self.config["dt"])
        self.fish_radius = float(self.config.get("fish_radius", 0.18))

        mpc_cfg = self.config.get("mpc", {})
        adaptive_cfg = mpc_cfg.get("adaptive_risk", {})
        self.horizon = int(mpc_cfg.get("N", 10))
        self.max_obstacles = int(mpc_cfg.get("max_obstacles", 10))
        self.obstacle_margin = float(mpc_cfg.get("obstacle_margin", 1.2))
        self.collision_margin = float(mpc_cfg.get("collision_margin", 0.2))
        self.heading_tau = float(mpc_cfg.get("heading_tau", 0.6))
        self.speed_tau = float(mpc_cfg.get("speed_tau", 0.3))
        self.max_theta_ref = float(mpc_cfg.get("max_theta_ref", np.deg2rad(35.0)))
        self.speed_min = float(mpc_cfg.get("speed_min", 0.04))
        self.speed_cap_floor = float(mpc_cfg.get("speed_cap_floor", 0.35))
        self.w_goal = float(mpc_cfg.get("W_goal", 18.0))
        self.w_terminal = float(mpc_cfg.get("W_terminal", 45.0))
        self.w_obs = float(mpc_cfg.get("W_obs", 220.0))
        self.w_collision = float(mpc_cfg.get("W_collision", 3000.0))
        self.w_speed = float(mpc_cfg.get("W_speed", 35.0))
        self.risk_speed_ref = float(mpc_cfg.get("risk_speed_ref", 0.25))
        self.risk_acc_ref = float(mpc_cfg.get("risk_acc_ref", 0.15))
        self.risk_size_ref = float(mpc_cfg.get("risk_size_ref", 1.0))
        self.risk_clearance_weight = float(adaptive_cfg.get("clearance_weight", 0.40))
        self.risk_speed_weight = float(adaptive_cfg.get("speed_weight", 0.30))
        self.risk_acc_weight = float(adaptive_cfg.get("acc_weight", 0.20))
        self.risk_size_weight = float(adaptive_cfg.get("size_weight", 0.10))
        self.obstacle_margin_gain = float(adaptive_cfg.get("obstacle_margin_gain", 1.2))
        self.obstacle_margin_scale_max = float(adaptive_cfg.get("obstacle_margin_scale_max", 2.2))
        self.collision_margin_gain = float(adaptive_cfg.get("collision_margin_gain", 1.5))
        self.collision_margin_scale_max = float(adaptive_cfg.get("collision_margin_scale_max", 2.5))
        self.w_obs_gain = float(adaptive_cfg.get("w_obs_gain", 2.0))
        self.w_obs_scale_max = float(adaptive_cfg.get("w_obs_scale_max", 3.0))
        self.w_collision_gain = float(adaptive_cfg.get("w_collision_gain", 3.0))
        self.w_collision_scale_max = float(adaptive_cfg.get("w_collision_scale_max", 4.0))
        self.speed_risk_slowdown = float(adaptive_cfg.get("speed_risk_slowdown", 0.65))
        self.track_stale_steps = int(mpc_cfg.get("track_stale_steps", 5))

        self._solver: dict[str, object] | None = None
        self._obstacle_history: dict[object, dict[str, np.ndarray | int]] = {}
        self._prev_robot_velocity_world: np.ndarray | None = None
        self._plan_step = 0
        self._warm_start: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._obstacle_history.clear()
        self._prev_robot_velocity_world = None
        self._plan_step = 0
        self._warm_start.clear()

    def _rotation_body_to_world(self, fish_state: np.ndarray) -> np.ndarray:
        fish_state = np.asarray(fish_state, dtype=np.float64)
        return quaternion_to_rotation_matrix(
            fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
        )

    def _world_velocity(self, fish_state: np.ndarray) -> np.ndarray:
        fish_state = np.asarray(fish_state, dtype=np.float64).reshape(-1)
        rotation = self._rotation_body_to_world(fish_state)
        return rotate_body_to_world(fish_state[[VX, VY, VZ]], rotation)

    def _build_solver(self) -> dict[str, object]:
        opti = ca.Opti()

        p0 = opti.parameter(POS_DIM, 1)
        psi0 = opti.parameter(1, 1)
        theta0 = opti.parameter(1, 1)
        v_prev_world = opti.parameter(POS_DIM, 1)
        speed0 = opti.parameter(1, 1)
        p_goal = opti.parameter(POS_DIM, 1)
        cmd_speed_nominal = opti.parameter(1, 1)
        obs_c = opti.parameter(POS_DIM, self.max_obstacles)
        obs_v = opti.parameter(POS_DIM, self.max_obstacles)
        obs_a = opti.parameter(POS_DIM, self.max_obstacles)
        obs_r = opti.parameter(1, self.max_obstacles)
        num_obs = opti.parameter(1, 1)
        w_goal_param = opti.parameter(1, 1)
        w_terminal_param = opti.parameter(1, 1)
        w_obs_param = opti.parameter(1, 1)
        w_collision_param = opti.parameter(1, 1)
        obstacle_margin_param = opti.parameter(1, 1)
        collision_margin_param = opti.parameter(1, 1)

        u = opti.variable(CONTROL_DIM, self.horizon)
        p = opti.variable(POS_DIM, self.horizon + 1)
        psi = opti.variable(1, self.horizon + 1)
        theta = opti.variable(1, self.horizon + 1)
        speed = opti.variable(1, self.horizon + 1)

        opti.subject_to(p[:, 0] == p0)
        opti.subject_to(psi[:, 0] == psi0)
        opti.subject_to(theta[:, 0] == theta0)
        opti.subject_to(speed[:, 0] == speed0)

        alpha = min(1.0, self.dt / max(self.heading_tau, 1e-6))
        beta = min(1.0, self.dt / max(self.speed_tau, 1e-6))
        nominal_speed_safe = ca.fmax(cmd_speed_nominal[0, 0], self.speed_min)
        speed_cap_high = ca.fmax(nominal_speed_safe, self.speed_cap_floor)
        cost = ca.MX(0.0)

        for k in range(self.horizon):
            psi_ref = u[0, k]
            theta_ref = u[1, k]
            speed_ref = u[2, k]
            p_k = p[:, k]
            psi_k = psi[0, k]
            theta_k = theta[0, k]
            speed_k = speed[0, k]

            psi_next = psi_k + alpha * _wrap_angle_mx(psi_ref - psi_k)
            theta_next = theta_k + alpha * (theta_ref - theta_k)
            speed_next = speed_k + beta * (speed_ref - speed_k)
            v_robot_prev = v_prev_world if k == 0 else _command_velocity(psi_k, theta_k, speed_k)
            v_robot_next = _command_velocity(psi_next, theta_next, speed_next)
            a_robot = (v_robot_next - v_robot_prev) / self.dt
            p_next = p_k + self.dt * v_robot_next

            opti.subject_to(psi[:, k + 1] == psi_next)
            opti.subject_to(theta[:, k + 1] == theta_next)
            opti.subject_to(speed[:, k + 1] == speed_next)
            opti.subject_to(p[:, k + 1] == p_next)

            cost += w_goal_param[0, 0] * ca.dot(p_next - p_goal, p_next - p_goal)
            risk_max_step = ca.MX(0.0)

            for obs_idx in range(self.max_obstacles):
                t_pred = (k + 1) * self.dt
                is_active = ca.if_else(num_obs[0, 0] > obs_idx, 1.0, 0.0)
                obs_p = (
                    obs_c[:, obs_idx]
                    + obs_v[:, obs_idx] * t_pred
                    + 0.5 * obs_a[:, obs_idx] * (t_pred * t_pred)
                )
                obs_vel = obs_v[:, obs_idx] + obs_a[:, obs_idx] * t_pred
                los_rel = obs_p - p_next
                dist = ca.sqrt(ca.dot(los_rel, los_rel) + 1e-6)
                clearance = dist - (obs_r[0, obs_idx] + self.fish_radius)
                n_hat = los_rel / dist
                rel_v = obs_vel - v_robot_next
                rel_a = obs_a[:, obs_idx] - a_robot
                closing_speed = ca.fmax(0.0, -ca.dot(rel_v, n_hat))
                closing_acc = ca.fmax(0.0, -ca.dot(rel_a, n_hat))
                clearance_term = _clip_mx(
                    (obstacle_margin_param[0, 0] - clearance) / ca.fmax(obstacle_margin_param[0, 0], 1e-6),
                    0.0,
                    1.0,
                )
                speed_term = _clip_mx(closing_speed / max(self.risk_speed_ref, 1e-6), 0.0, 1.0)
                acc_term = _clip_mx(closing_acc / max(self.risk_acc_ref, 1e-6), 0.0, 1.0)
                size_term = _clip_mx(obs_r[0, obs_idx] / max(self.risk_size_ref, 1e-6), 0.0, 1.0)
                risk = _clip_mx(
                    self.risk_clearance_weight * clearance_term
                    + self.risk_speed_weight * speed_term
                    + self.risk_acc_weight * acc_term
                    + self.risk_size_weight * size_term,
                    0.0,
                    1.0,
                )

                adaptive_obstacle_margin = ca.fmin(
                    obstacle_margin_param[0, 0] * (1.0 + self.obstacle_margin_gain * risk),
                    obstacle_margin_param[0, 0] * self.obstacle_margin_scale_max,
                )
                adaptive_collision_margin = ca.fmin(
                    collision_margin_param[0, 0] * (1.0 + self.collision_margin_gain * risk),
                    collision_margin_param[0, 0] * self.collision_margin_scale_max,
                )
                adaptive_w_obs = ca.fmin(
                    w_obs_param[0, 0] * (1.0 + self.w_obs_gain * risk),
                    w_obs_param[0, 0] * self.w_obs_scale_max,
                )
                adaptive_w_collision = ca.fmin(
                    w_collision_param[0, 0] * (1.0 + self.w_collision_gain * risk),
                    w_collision_param[0, 0] * self.w_collision_scale_max,
                )

                near_pen = _smooth_positive_part(adaptive_obstacle_margin - clearance)
                collision_pen = _smooth_positive_part(adaptive_collision_margin - clearance)
                collision_overlap = _smooth_positive_part(-clearance)
                obs_cost = (
                    adaptive_w_obs * near_pen * near_pen * near_pen
                    + adaptive_w_collision * collision_pen * collision_pen
                    + 10.0 * adaptive_w_collision * collision_overlap * collision_overlap
                )
                cost += is_active * obs_cost
                risk_max_step = ca.fmax(risk_max_step, is_active * risk)

            speed_des = _clip_mx(
                nominal_speed_safe * (1.0 - self.speed_risk_slowdown * risk_max_step),
                self.speed_min,
                nominal_speed_safe,
            )
            speed_cap_step = nominal_speed_safe + risk_max_step * (speed_cap_high - nominal_speed_safe)
            cost += self.w_speed * (speed_ref - speed_des) * (speed_ref - speed_des)

            opti.subject_to(opti.bounded(-self.max_theta_ref, theta_ref, self.max_theta_ref))
            opti.subject_to(opti.bounded(-self.max_theta_ref, theta_next, self.max_theta_ref))
            opti.subject_to(speed_ref >= self.speed_min)
            opti.subject_to(speed_ref <= speed_cap_step)
            opti.subject_to(speed_next >= self.speed_min)
            opti.subject_to(speed_next <= speed_cap_step)

        cost += w_terminal_param[0, 0] * ca.dot(p[:, self.horizon] - p_goal, p[:, self.horizon] - p_goal)
        opti.minimize(cost)

        opti.set_initial(p, np.zeros((POS_DIM, self.horizon + 1), dtype=np.float64))
        opti.set_initial(psi, np.zeros((1, self.horizon + 1), dtype=np.float64))
        opti.set_initial(theta, np.zeros((1, self.horizon + 1), dtype=np.float64))
        opti.set_initial(speed, np.full((1, self.horizon + 1), self.speed_min, dtype=np.float64))
        opti.set_initial(u, np.zeros((CONTROL_DIM, self.horizon), dtype=np.float64))
        opti.solver(
            "ipopt",
            {"print_time": False},
            {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 100,
                "tol": 1e-3,
                "linear_solver": "mumps",
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-6,
                "warm_start_mult_bound_push": 1e-6,
                "warm_start_slack_bound_push": 1e-6,
            },
        )
        return {
            "opti": opti,
            "params": {
                "p0": p0,
                "psi0": psi0,
                "theta0": theta0,
                "v_prev_world": v_prev_world,
                "speed0": speed0,
                "p_goal": p_goal,
                "cmd_speed_nominal": cmd_speed_nominal,
                "obs_c": obs_c,
                "obs_v": obs_v,
                "obs_a": obs_a,
                "obs_r": obs_r,
                "num_obs": num_obs,
                "w_goal_param": w_goal_param,
                "w_terminal_param": w_terminal_param,
                "w_obs_param": w_obs_param,
                "w_collision_param": w_collision_param,
                "obstacle_margin_param": obstacle_margin_param,
                "collision_margin_param": collision_margin_param,
            },
            "vars": {
                "u": u,
                "p": p,
                "psi": psi,
                "theta": theta,
                "speed": speed,
                "cost": cost,
            },
        }

    def _ensure_solver(self) -> dict[str, object]:
        if self._solver is None:
            self._solver = self._build_solver()
        return self._solver

    @staticmethod
    def _shift_columns(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] <= 1:
            return values.copy()
        return np.concatenate([values[:, 1:], values[:, -1:]], axis=1)

    def _set_default_initial_guess(
        self,
        problem: dict[str, object],
        p0: np.ndarray,
        psi0: float,
        theta0: float,
        speed0: float,
        p_goal: np.ndarray,
        cmd_speed_nominal: float,
    ) -> None:
        opti = problem["opti"]
        vars_dict = problem["vars"]

        del p0, psi0, theta0, speed0, p_goal, cmd_speed_nominal
        opti.set_initial(vars_dict["p"], np.zeros((POS_DIM, self.horizon + 1), dtype=np.float64))
        opti.set_initial(vars_dict["psi"], np.zeros((1, self.horizon + 1), dtype=np.float64))
        opti.set_initial(vars_dict["theta"], np.zeros((1, self.horizon + 1), dtype=np.float64))
        opti.set_initial(vars_dict["speed"], np.full((1, self.horizon + 1), self.speed_min, dtype=np.float64))
        opti.set_initial(vars_dict["u"], np.zeros((CONTROL_DIM, self.horizon), dtype=np.float64))

    def _apply_warm_start(self, problem: dict[str, object]) -> None:
        if not self._warm_start:
            return
        vars_dict = problem["vars"]
        for name in ("u", "p", "psi", "theta", "speed"):
            values = self._warm_start.get(name)
            if values is not None:
                problem["opti"].set_initial(vars_dict[name], values)

    def _store_warm_start(
        self,
        u_seq: np.ndarray,
        p_seq: np.ndarray,
        psi_seq: np.ndarray,
        theta_seq: np.ndarray,
        speed_seq: np.ndarray,
    ) -> None:
        self._warm_start = {
            "u": self._shift_columns(np.asarray(u_seq, dtype=np.float64)),
            "p": self._shift_columns(np.asarray(p_seq, dtype=np.float64)),
            "psi": self._shift_columns(np.asarray(psi_seq, dtype=np.float64).reshape(1, -1)),
            "theta": self._shift_columns(np.asarray(theta_seq, dtype=np.float64).reshape(1, -1)),
            "speed": self._shift_columns(np.asarray(speed_seq, dtype=np.float64).reshape(1, -1)),
        }

    def _pack_obstacles(
        self,
        visible_obstacles: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list[object]]:
        obs_c = np.zeros((POS_DIM, self.max_obstacles), dtype=np.float64)
        obs_v = np.zeros((POS_DIM, self.max_obstacles), dtype=np.float64)
        obs_a = np.zeros((POS_DIM, self.max_obstacles), dtype=np.float64)
        obs_r = np.zeros((1, self.max_obstacles), dtype=np.float64)
        track_keys: list[object] = []
        num_obs = min(len(visible_obstacles), self.max_obstacles)
        for idx in range(num_obs):
            obstacle = visible_obstacles[idx]
            track_key = obstacle.get("track_id", id(obstacle))
            center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
            velocity = np.asarray(obstacle.get("v", np.zeros(3)), dtype=np.float64).reshape(3)
            previous = self._obstacle_history.get(track_key)
            acceleration = np.zeros(3, dtype=np.float64)
            if previous is not None:
                acceleration = (velocity - np.asarray(previous["velocity"], dtype=np.float64)) / self.dt
            self._obstacle_history[track_key] = {
                "center": center.copy(),
                "velocity": velocity.copy(),
                "step_idx": int(self._plan_step),
            }
            obs_c[:, idx] = center
            obs_v[:, idx] = velocity
            obs_a[:, idx] = acceleration
            obs_r[0, idx] = float(obstacle["r"])
            track_keys.append(track_key)

        stale_before = self._plan_step - self.track_stale_steps
        for track_key in list(self._obstacle_history):
            if int(self._obstacle_history[track_key]["step_idx"]) < stale_before:
                del self._obstacle_history[track_key]

        return obs_c, obs_v, obs_a, obs_r, float(num_obs), track_keys

    def _risk_value(
        self,
        clearance: float,
        closing_speed: float,
        closing_acc: float,
        obstacle_radius: float,
        obstacle_margin: float,
    ) -> float:
        clearance_term = np.clip(
            (float(obstacle_margin) - clearance) / max(float(obstacle_margin), 1e-6),
            0.0,
            1.0,
        )
        speed_term = np.clip(closing_speed / max(self.risk_speed_ref, 1e-6), 0.0, 1.0)
        acc_term = np.clip(closing_acc / max(self.risk_acc_ref, 1e-6), 0.0, 1.0)
        size_term = np.clip(obstacle_radius / max(self.risk_size_ref, 1e-6), 0.0, 1.0)
        return float(
            np.clip(
                self.risk_clearance_weight * clearance_term
                + self.risk_speed_weight * speed_term
                + self.risk_acc_weight * acc_term
                + self.risk_size_weight * size_term,
                0.0,
                1.0,
            )
        )

    def _risk_diagnostics(
        self,
        p_seq: np.ndarray,
        psi_seq: np.ndarray,
        theta_seq: np.ndarray,
        speed_seq: np.ndarray,
        current_velocity_world: np.ndarray,
        previous_velocity_world: np.ndarray,
        obs_c: np.ndarray,
        obs_v: np.ndarray,
        obs_a: np.ndarray,
        obs_r: np.ndarray,
        num_obs: int,
        obstacle_margin: float,
        collision_margin: float,
    ) -> dict:
        current_velocity_world = np.asarray(current_velocity_world, dtype=np.float64).reshape(3)
        previous_velocity_world = np.asarray(previous_velocity_world, dtype=np.float64).reshape(3)
        obs_c = np.asarray(obs_c, dtype=np.float64)
        obs_v = np.asarray(obs_v, dtype=np.float64)
        obs_a = np.asarray(obs_a, dtype=np.float64)
        obs_r = np.asarray(obs_r, dtype=np.float64)
        p_seq = np.asarray(p_seq, dtype=np.float64)
        psi_seq = np.asarray(psi_seq, dtype=np.float64).reshape(-1)
        theta_seq = np.asarray(theta_seq, dtype=np.float64).reshape(-1)
        speed_seq = np.asarray(speed_seq, dtype=np.float64).reshape(-1)

        vel_seq = np.zeros((POS_DIM, speed_seq.shape[0]), dtype=np.float64)
        vel_seq[:, 0] = current_velocity_world
        for idx in range(1, speed_seq.shape[0]):
            vel_seq[:, idx] = np.array(
                [
                    speed_seq[idx] * np.cos(theta_seq[idx]) * np.cos(psi_seq[idx]),
                    speed_seq[idx] * np.cos(theta_seq[idx]) * np.sin(psi_seq[idx]),
                    -speed_seq[idx] * np.sin(theta_seq[idx]),
                ],
                dtype=np.float64,
            )

        risk_per_obstacle = np.zeros(num_obs, dtype=np.float64)
        risk_max = 0.0
        adaptive_margin_max = float(obstacle_margin)
        adaptive_collision_margin_max = float(collision_margin)

        for step_idx in range(1, p_seq.shape[1]):
            prev_velocity = previous_velocity_world if step_idx == 1 else vel_seq[:, step_idx - 1]
            robot_acc = (vel_seq[:, step_idx] - prev_velocity) / self.dt
            t_pred = step_idx * self.dt
            for obs_idx in range(num_obs):
                obs_pos = obs_c[:, obs_idx] + obs_v[:, obs_idx] * t_pred + 0.5 * obs_a[:, obs_idx] * (t_pred**2)
                obs_vel = obs_v[:, obs_idx] + obs_a[:, obs_idx] * t_pred
                los_rel = obs_pos - p_seq[:, step_idx]
                dist = float(np.linalg.norm(los_rel))
                if dist < 1e-6:
                    n_hat = np.zeros(3, dtype=np.float64)
                else:
                    n_hat = los_rel / dist
                clearance = dist - (float(obs_r[0, obs_idx]) + self.fish_radius)
                rel_v = obs_vel - vel_seq[:, step_idx]
                rel_a = obs_a[:, obs_idx] - robot_acc
                closing_speed = max(0.0, -float(np.dot(rel_v, n_hat)))
                closing_acc = max(0.0, -float(np.dot(rel_a, n_hat)))
                risk = self._risk_value(
                    clearance,
                    closing_speed,
                    closing_acc,
                    float(obs_r[0, obs_idx]),
                    obstacle_margin,
                )
                risk_per_obstacle[obs_idx] = max(risk_per_obstacle[obs_idx], risk)
                risk_max = max(risk_max, risk)
                adaptive_margin_max = max(
                    adaptive_margin_max,
                    min(
                        obstacle_margin * (1.0 + self.obstacle_margin_gain * risk),
                        obstacle_margin * self.obstacle_margin_scale_max,
                    ),
                )
                adaptive_collision_margin_max = max(
                    adaptive_collision_margin_max,
                    min(
                        collision_margin * (1.0 + self.collision_margin_gain * risk),
                        collision_margin * self.collision_margin_scale_max,
                    ),
                )

        return {
            "risk_max": float(risk_max),
            "risk_per_obstacle": risk_per_obstacle,
            "adaptive_margin_max": float(adaptive_margin_max),
            "adaptive_collision_margin_max": float(adaptive_collision_margin_max),
        }

    def _resolve_strategy_params(self, strategy_params: dict | None) -> dict:
        params = {
            "mode_name": "mpc",
            "speed_scale": 1.0,
            "W_goal": self.w_goal,
            "W_terminal": self.w_terminal,
            "W_obs": self.w_obs,
            "W_collision": self.w_collision,
            "obstacle_margin": self.obstacle_margin,
            "collision_margin": self.collision_margin,
        }
        if strategy_params:
            for key, value in strategy_params.items():
                if key in params:
                    params[key] = float(value) if key != "mode_name" else value
                elif key == "mode_name":
                    params["mode_name"] = str(value)
        params["mode_name"] = str(params["mode_name"])
        return params

    def plan(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
        cmd_speed: float,
        strategy_params: dict | None = None,
    ) -> tuple[np.ndarray, float, dict]:
        del hist
        problem = self._ensure_solver()
        opti = problem["opti"]
        params = problem["params"]
        vars_dict = problem["vars"]
        active_strategy = self._resolve_strategy_params(strategy_params)

        fish_state = np.asarray(fish_state, dtype=np.float64).reshape(-1)
        p0 = fish_state[[PX, PY, PZ]].reshape(3, 1)
        psi0, theta0 = extract_attitude(fish_state)
        current_velocity_world = self._world_velocity(fish_state)
        previous_velocity_world = (
            current_velocity_world.copy()
            if self._prev_robot_velocity_world is None
            else np.asarray(self._prev_robot_velocity_world, dtype=np.float64).reshape(3)
        )
        speed0 = float(np.linalg.norm(current_velocity_world))

        p_goal = np.asarray(local_target, dtype=np.float64).reshape(3, 1)
        obs_c, obs_v, obs_a, obs_r, num_obs, track_keys = self._pack_obstacles(visible_obstacles)
        cmd_speed_nominal = max(self.speed_min, float(cmd_speed) * float(active_strategy["speed_scale"]))

        try:
            opti.set_value(params["p0"], p0)
            opti.set_value(params["psi0"], np.array([[psi0]], dtype=np.float64))
            opti.set_value(params["theta0"], np.array([[theta0]], dtype=np.float64))
            opti.set_value(params["v_prev_world"], previous_velocity_world.reshape(3, 1))
            opti.set_value(params["speed0"], np.array([[speed0]], dtype=np.float64))
            opti.set_value(params["p_goal"], p_goal)
            opti.set_value(params["cmd_speed_nominal"], np.array([[cmd_speed_nominal]], dtype=np.float64))
            opti.set_value(params["obs_c"], obs_c)
            opti.set_value(params["obs_v"], obs_v)
            opti.set_value(params["obs_a"], obs_a)
            opti.set_value(params["obs_r"], obs_r)
            opti.set_value(params["num_obs"], np.array([[num_obs]], dtype=np.float64))
            opti.set_value(params["w_goal_param"], np.array([[float(active_strategy["W_goal"])]], dtype=np.float64))
            opti.set_value(
                params["w_terminal_param"],
                np.array([[float(active_strategy["W_terminal"])]], dtype=np.float64),
            )
            opti.set_value(params["w_obs_param"], np.array([[float(active_strategy["W_obs"])]], dtype=np.float64))
            opti.set_value(
                params["w_collision_param"],
                np.array([[float(active_strategy["W_collision"])]], dtype=np.float64),
            )
            opti.set_value(
                params["obstacle_margin_param"],
                np.array([[float(active_strategy["obstacle_margin"])]], dtype=np.float64),
            )
            opti.set_value(
                params["collision_margin_param"],
                np.array([[float(active_strategy["collision_margin"])]], dtype=np.float64),
            )
            self._set_default_initial_guess(problem, p0, psi0, theta0, speed0, p_goal, cmd_speed_nominal)
            self._apply_warm_start(problem)

            solution = opti.solve_limited()
            stats = opti.stats()
            u_seq = np.asarray(solution.value(vars_dict["u"]), dtype=np.float64)
            p_seq = np.asarray(solution.value(vars_dict["p"]), dtype=np.float64)
            psi_seq = np.asarray(solution.value(vars_dict["psi"]), dtype=np.float64).reshape(-1)
            theta_seq = np.asarray(solution.value(vars_dict["theta"]), dtype=np.float64).reshape(-1)
            speed_seq = np.asarray(solution.value(vars_dict["speed"]), dtype=np.float64).reshape(-1)
            total_cost = float(np.asarray(solution.value(vars_dict["cost"]), dtype=np.float64).reshape(-1)[0])
            self._store_warm_start(
                u_seq,
                p_seq,
                psi_seq,
                theta_seq,
                speed_seq,
            )
            attitude_ref = np.array([u_seq[0, 0], u_seq[1, 0]], dtype=np.float64)
            cmd_speed_ref = float(u_seq[2, 0])
            cmd_vel_global = np.array(
                [
                    cmd_speed_ref * np.cos(attitude_ref[1]) * np.cos(attitude_ref[0]),
                    cmd_speed_ref * np.cos(attitude_ref[1]) * np.sin(attitude_ref[0]),
                    -cmd_speed_ref * np.sin(attitude_ref[1]),
                ],
                dtype=np.float64,
            )
            diagnostics = self._risk_diagnostics(
                p_seq,
                psi_seq,
                theta_seq,
                speed_seq,
                current_velocity_world,
                previous_velocity_world,
                obs_c,
                obs_v,
                obs_a,
                obs_r,
                int(num_obs),
                float(active_strategy["obstacle_margin"]),
                float(active_strategy["collision_margin"]),
            )
            planner_info = {
                "success": bool(
                    stats.get("success", False)
                    or stats.get("return_status") in {"Maximum_Iterations_Exceeded", "Solved_To_Acceptable_Level"}
                ),
                "cost": total_cost,
                "local_target": p_goal.reshape(-1).copy(),
                "psi_ref": float(attitude_ref[0]),
                "theta_ref": float(attitude_ref[1]),
                "cmd_speed_ref": cmd_speed_ref,
                "cmd_vel_global": cmd_vel_global,
                "u_seq": u_seq,
                "p_seq": p_seq,
                "psi_seq": psi_seq,
                "theta_seq": theta_seq,
                "speed_seq": speed_seq,
                "track_keys": track_keys,
                "mode_name": active_strategy["mode_name"],
                "strategy_params": active_strategy.copy(),
                "solver_status": stats.get("return_status"),
            }
            planner_info.update(diagnostics)
            return attitude_ref, cmd_speed_ref, planner_info
        except Exception as exc:
            self._warm_start.clear()
            fallback = np.array([psi0, theta0], dtype=np.float64)
            fallback_speed = float(max(0.7 * cmd_speed_nominal, self.speed_min))
            cmd_vel_global = np.array(
                [
                    fallback_speed * np.cos(theta0) * np.cos(psi0),
                    fallback_speed * np.cos(theta0) * np.sin(psi0),
                    -fallback_speed * np.sin(theta0),
                ],
                dtype=np.float64,
            )
            return fallback, fallback_speed, {
                "success": False,
                "error": str(exc),
                "local_target": p_goal.reshape(-1).copy(),
                "psi_ref": float(psi0),
                "theta_ref": float(theta0),
                "cmd_speed_ref": fallback_speed,
                "cmd_vel_global": cmd_vel_global,
                "speed_seq": np.array([fallback_speed], dtype=np.float64),
                "risk_max": 0.0,
                "risk_per_obstacle": np.zeros(int(num_obs), dtype=np.float64),
                "adaptive_margin_max": float(active_strategy["obstacle_margin"]),
                "adaptive_collision_margin_max": float(active_strategy["collision_margin"]),
                "track_keys": track_keys,
                "mode_name": active_strategy["mode_name"],
                "strategy_params": active_strategy.copy(),
            }
        finally:
            self._prev_robot_velocity_world = current_velocity_world.copy()
            self._plan_step += 1
