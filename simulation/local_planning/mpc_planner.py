from __future__ import annotations

import casadi as ca
import numpy as np

from dynamics_wrapper.indices import PX, PY, PZ
from rl.configs.fish_env import build_fish_env_config
from simulation.local_planning.fin_controller import extract_attitude


POS_DIM = 3
REF_DIM = 2


def _smooth_positive_part(value: ca.MX) -> ca.MX:
    return 0.5 * (value + ca.sqrt(value * value + 1e-6))


def _wrap_angle_mx(angle: ca.MX) -> ca.MX:
    return ca.atan2(ca.sin(angle), ca.cos(angle))


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
        self.horizon = int(mpc_cfg.get("N", 10))
        self.max_obstacles = int(mpc_cfg.get("max_obstacles", 10))
        self.obstacle_margin = float(mpc_cfg.get("obstacle_margin", 1.2))
        self.collision_margin = float(mpc_cfg.get("collision_margin", 0.2))
        self.heading_tau = float(mpc_cfg.get("heading_tau", 0.6))
        self.max_theta_ref = float(mpc_cfg.get("max_theta_ref", np.deg2rad(35.0)))
        self.w_goal = float(mpc_cfg.get("W_goal", 18.0))
        self.w_terminal = float(mpc_cfg.get("W_terminal", 45.0))
        self.w_obs = float(mpc_cfg.get("W_obs", 220.0))
        self.w_collision = float(mpc_cfg.get("W_collision", 3000.0))

        self._solver: ca.Function | None = None

    def reset(self) -> None:
        return None

    def _build_solver(self) -> ca.Function:
        opti = ca.Opti()

        p0 = opti.parameter(POS_DIM, 1)
        psi0 = opti.parameter(1, 1)
        theta0 = opti.parameter(1, 1)
        p_goal = opti.parameter(POS_DIM, 1)
        cmd_speed = opti.parameter(1, 1)
        obs_c = opti.parameter(POS_DIM, self.max_obstacles)
        obs_v = opti.parameter(POS_DIM, self.max_obstacles)
        obs_r = opti.parameter(1, self.max_obstacles)
        num_obs = opti.parameter(1, 1)

        u = opti.variable(REF_DIM, self.horizon)
        p = opti.variable(POS_DIM, self.horizon + 1)
        psi = opti.variable(1, self.horizon + 1)
        theta = opti.variable(1, self.horizon + 1)

        opti.subject_to(p[:, 0] == p0)
        opti.subject_to(psi[:, 0] == psi0)
        opti.subject_to(theta[:, 0] == theta0)

        alpha = min(1.0, self.dt / max(self.heading_tau, 1e-6))
        cost = ca.MX(0.0)

        for k in range(self.horizon):
            psi_ref = u[0, k]
            theta_ref = u[1, k]
            p_k = p[:, k]
            psi_k = psi[0, k]
            theta_k = theta[0, k]

            psi_next = psi_k + alpha * _wrap_angle_mx(psi_ref - psi_k)
            theta_next = theta_k + alpha * (theta_ref - theta_k)
            p_next = p_k + self.dt * _command_velocity(psi_next, theta_next, cmd_speed[0, 0])

            opti.subject_to(psi[:, k + 1] == psi_next)
            opti.subject_to(theta[:, k + 1] == theta_next)
            opti.subject_to(p[:, k + 1] == p_next)

            cost += self.w_goal * ca.dot(p_next - p_goal, p_next - p_goal)

            for obs_idx in range(self.max_obstacles):
                obs_p = obs_c[:, obs_idx] + obs_v[:, obs_idx] * ((k + 1) * self.dt)
                rel = p_next - obs_p
                clearance = ca.sqrt(ca.dot(rel, rel) + 1e-6) - (obs_r[0, obs_idx] + self.fish_radius)
                near_pen = _smooth_positive_part(self.obstacle_margin - clearance)
                collision_pen = _smooth_positive_part(self.collision_margin - clearance)
                collision_overlap = _smooth_positive_part(-clearance)
                obs_cost = (
                    self.w_obs * near_pen * near_pen * near_pen
                    + self.w_collision * collision_pen * collision_pen
                    + 10.0 * self.w_collision * collision_overlap * collision_overlap
                )
                cost += ca.if_else(num_obs > obs_idx, obs_cost, 0.0)

            opti.subject_to(opti.bounded(-self.max_theta_ref, theta_ref, self.max_theta_ref))
            opti.subject_to(opti.bounded(-self.max_theta_ref, theta_next, self.max_theta_ref))

        cost += self.w_terminal * ca.dot(p[:, self.horizon] - p_goal, p[:, self.horizon] - p_goal)
        opti.minimize(cost)

        opti.set_initial(p, np.zeros((POS_DIM, self.horizon + 1), dtype=np.float64))
        opti.set_initial(psi, np.zeros((1, self.horizon + 1), dtype=np.float64))
        opti.set_initial(theta, np.zeros((1, self.horizon + 1), dtype=np.float64))
        opti.set_initial(u, np.zeros((REF_DIM, self.horizon), dtype=np.float64))
        opti.solver(
            "ipopt",
            {"print_time": False},
            {"print_level": 0, "sb": "yes", "max_iter": 100, "tol": 1e-3, "linear_solver": "mumps"},
        )
        return opti.to_function(
            "attitude_mpc_solver",
            [p0, psi0, theta0, p_goal, cmd_speed, obs_c, obs_v, obs_r, num_obs],
            [u, p, psi, theta, cost],
        )

    def _ensure_solver(self) -> ca.Function:
        if self._solver is None:
            self._solver = self._build_solver()
        return self._solver

    def _pack_obstacles(
        self,
        visible_obstacles: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        obs_c = np.zeros((POS_DIM, self.max_obstacles), dtype=np.float64)
        obs_v = np.zeros((POS_DIM, self.max_obstacles), dtype=np.float64)
        obs_r = np.zeros((1, self.max_obstacles), dtype=np.float64)
        num_obs = min(len(visible_obstacles), self.max_obstacles)
        for idx in range(num_obs):
            obstacle = visible_obstacles[idx]
            obs_c[:, idx] = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
            obs_v[:, idx] = np.asarray(obstacle.get("v", np.zeros(3)), dtype=np.float64).reshape(3)
            obs_r[0, idx] = float(obstacle["r"])
        return obs_c, obs_v, obs_r, float(num_obs)

    def plan(
        self,
        fish_state: np.ndarray,
        hist: np.ndarray,
        local_target: np.ndarray,
        visible_obstacles: list[dict],
        cmd_speed: float,
    ) -> tuple[np.ndarray, dict]:
        del hist
        solver = self._ensure_solver()

        fish_state = np.asarray(fish_state, dtype=np.float64).reshape(-1)
        p0 = fish_state[[PX, PY, PZ]].reshape(3, 1)
        psi0, theta0 = extract_attitude(fish_state)

        p_goal = np.asarray(local_target, dtype=np.float64).reshape(3, 1)
        obs_c, obs_v, obs_r, num_obs = self._pack_obstacles(visible_obstacles)

        try:
            u_opt, p_opt, psi_opt, theta_opt, total_cost = solver(
                p0,
                np.array([[psi0]], dtype=np.float64),
                np.array([[theta0]], dtype=np.float64),
                p_goal,
                np.array([[float(cmd_speed)]], dtype=np.float64),
                obs_c,
                obs_v,
                obs_r,
                np.array([[num_obs]], dtype=np.float64),
            )
            u_seq = np.asarray(u_opt, dtype=np.float64)
            p_seq = np.asarray(p_opt, dtype=np.float64)
            psi_seq = np.asarray(psi_opt, dtype=np.float64).reshape(-1)
            theta_seq = np.asarray(theta_opt, dtype=np.float64).reshape(-1)
            attitude_ref = np.array([u_seq[0, 0], u_seq[1, 0]], dtype=np.float64)
            cmd_vel_global = np.array(
                [
                    cmd_speed * np.cos(attitude_ref[1]) * np.cos(attitude_ref[0]),
                    cmd_speed * np.cos(attitude_ref[1]) * np.sin(attitude_ref[0]),
                    -cmd_speed * np.sin(attitude_ref[1]),
                ],
                dtype=np.float64,
            )
            return attitude_ref, {
                "success": True,
                "cost": float(np.asarray(total_cost).reshape(-1)[0]),
                "local_target": p_goal.reshape(-1).copy(),
                "psi_ref": float(attitude_ref[0]),
                "theta_ref": float(attitude_ref[1]),
                "cmd_vel_global": cmd_vel_global,
                "u_seq": u_seq,
                "p_seq": p_seq,
                "psi_seq": psi_seq,
                "theta_seq": theta_seq,
            }
        except Exception as exc:
            fallback = np.array([psi0, theta0], dtype=np.float64)
            cmd_vel_global = np.array(
                [
                    cmd_speed * np.cos(theta0) * np.cos(psi0),
                    cmd_speed * np.cos(theta0) * np.sin(psi0),
                    -cmd_speed * np.sin(theta0),
                ],
                dtype=np.float64,
            )
            return fallback, {
                "success": False,
                "error": str(exc),
                "local_target": p_goal.reshape(-1).copy(),
                "psi_ref": float(psi0),
                "theta_ref": float(theta0),
                "cmd_vel_global": cmd_vel_global,
            }
