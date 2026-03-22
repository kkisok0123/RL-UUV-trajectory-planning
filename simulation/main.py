from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

# Allow `python simulation/main.py` from the repo root or an IDE run config.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dynamics_wrapper import step as dynamics_step
from dynamics_wrapper.params import load_body_params, load_fin_params
from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3
from rl.local_planner import RLLocalPlanner
from rl.configs.fish_env import build_fish_env_config
from simulation.global_planning.los import los_guidance_3d
# from simulation.global_planning.plan_global_bezier_pso import planGlobalBezierPSO
from simulation.local_planning.fin_controller import fin_controller
from simulation.local_planning.path_utils import find_local_target
from simulation.local_planning.sensor import get_visible_obstacles


@dataclass
class HybridSimulationConfig:
    dt: float = 0.1
    max_steps: int = 2500
    goal_threshold: float = 1.0
    local_lookahead: float = 4.0


def _make_obstacle(center, radius, velocity=None) -> dict:
    if velocity is None:
        velocity = np.zeros(3, dtype=np.float64)
    return {
        "c": np.asarray(center, dtype=np.float64).reshape(3),
        "r": float(radius),
        "v": np.asarray(velocity, dtype=np.float64).reshape(3),
    }


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
    return quat / np.linalg.norm(quat)


def _catmull_rom_chain(waypoints: np.ndarray, samples_per_segment: int) -> np.ndarray:
    pts = np.asarray(waypoints, dtype=np.float64)
    tangents = np.zeros_like(pts)
    tangents[0] = 0.5 * (pts[1] - pts[0])
    tangents[-1] = 0.5 * (pts[-1] - pts[-2])
    tangents[1:-1] = 0.5 * (pts[2:] - pts[:-2])

    samples: list[np.ndarray] = []
    for idx in range(len(pts) - 1):
        p0 = pts[idx]
        p1 = pts[idx + 1]
        m0 = tangents[idx]
        m1 = tangents[idx + 1]
        for tau in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            h00 = 2.0 * tau**3 - 3.0 * tau**2 + 1.0
            h10 = tau**3 - 2.0 * tau**2 + tau
            h01 = -2.0 * tau**3 + 3.0 * tau**2
            h11 = tau**3 - tau**2
            samples.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    samples.append(pts[-1].copy())
    return np.asarray(samples, dtype=np.float64)


def _path_tangent(path_xyz: np.ndarray, index: int) -> np.ndarray:
    path_xyz = np.asarray(path_xyz, dtype=np.float64)
    idx = int(np.clip(index, 0, path_xyz.shape[0] - 1))
    if path_xyz.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if idx == 0:
        tangent = path_xyz[1] - path_xyz[0]
    elif idx == path_xyz.shape[0] - 1:
        tangent = path_xyz[-1] - path_xyz[-2]
    else:
        tangent = 0.5 * (path_xyz[idx + 1] - path_xyz[idx - 1])
    if np.linalg.norm(tangent) < 1e-8:
        tangent = path_xyz[min(idx + 1, path_xyz.shape[0] - 1)] - path_xyz[max(idx - 1, 0)]
    return np.asarray(tangent, dtype=np.float64)


def _compute_tracking_metrics(settings: dict, controller_params: dict, steps_executed: int) -> dict:
    if steps_executed <= 0:
        return {}

    theta_err = np.asarray(settings.get("e_theta", np.zeros(steps_executed)), dtype=np.float64)[:steps_executed]
    psi_err = np.asarray(settings.get("e_psi", np.zeros(steps_executed)), dtype=np.float64)[:steps_executed]
    alpha5_ref = np.asarray(settings.get("alpha5_ref", np.zeros(steps_executed)), dtype=np.float64)[
        :steps_executed
    ]
    delta_ref = np.asarray(settings.get("delta_ref", np.zeros(steps_executed)), dtype=np.float64)[
        :steps_executed
    ]

    alpha5_limit = max(abs(float(controller_params["alpha5_min"])), abs(float(controller_params["alpha5_max"])))
    delta_limit = abs(float(controller_params["delta_rot_max"]))
    sat_tol = np.deg2rad(0.5)

    return {
        "theta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(theta_err**2)))),
        "theta_mae_deg": float(np.rad2deg(np.mean(np.abs(theta_err)))),
        "psi_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(psi_err**2)))),
        "psi_mae_deg": float(np.rad2deg(np.mean(np.abs(psi_err)))),
        "alpha5_sat_pct": float(100.0 * np.mean(np.abs(np.abs(alpha5_ref) - alpha5_limit) <= sat_tol)),
        "delta_sat_pct": float(100.0 * np.mean(np.abs(np.abs(delta_ref) - delta_limit) <= sat_tol)),
    }


def _build_reference_path_from_tracking_template(start: np.ndarray, goal: np.ndarray) -> dict:
    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal = np.asarray(goal, dtype=np.float64).reshape(3)
    # Use explicit world-frame waypoints so the nominal route stays clear of the
    # current obstacle chain while preserving a steady climb toward the goal.
    waypoints = np.array(
        [
            start,
            [8.0, -36.0, -55.0],
            [16.0, -32.0, -49.0],
            [24.0, -27.0, -41.0],
            [32.0, -20.0, -32.0],
            [40.0, -10.0, -22.0],
            [48.0, 0.0, -12.0],
            [56.0, 12.0, -2.0],
            [64.0, 24.0, 8.0],
            [70.0, 36.0, 14.0],
            [75.0, 48.0, 18.0],
            goal,
        ],
        dtype=np.float64,
    )
    xyz = _catmull_rom_chain(waypoints, samples_per_segment=80)
    return {"waypoints": waypoints, "xyz": xyz}


def run_hybrid_los_rl_simulation(
    model_path=None,
    visualize: bool = False,
    controller_params_override: dict | None = None,
) -> dict:
    cfg = HybridSimulationConfig()
    env_cfg = build_fish_env_config()
    sa_settings: dict[str, np.ndarray] = {}

    fs = {"p": np.array([0.0, -40.0, -60.0]), "v": np.array([1.2, 1.5, 1.2]), "a": np.zeros(3)}
    fg = {"p": np.array([80.0, 60.0, 20.0]), "v": np.zeros(3), "a": np.zeros(3)}
    scene_min = np.array([0.0, -40.0, -60.0], dtype=np.float64)
    scene_max = np.array([80.0, 60.0, 20.0], dtype=np.float64)

    cruise_speed = 1.2
    # opts = {
    #     "t0": 0.0,
    #     "t3": 60.0,
    #     "vmax": 1.2,
    #     "amax": 1.5,
    #     "N": 500,
    #     "psoM": 200,
    #     "psoT": 400,
    #     "w": 0.7,
    #     "c1": 1.6,
    #     "c2": 1.6,
    #     "clearance": 0.2,
    #     "Mk_obs": 1e3,
    #     "Mk_speed": 1e2,
    #     "Mk_acc": 1e2,
    #     "Mk_kappa": 1e2,
    #     "dt": cfg.dt,
    # }

    los_params = {"Delta": 2.5, "k_p": 0.5}
    static_obs = [
        _make_obstacle([16.0, -22.0, -44.0], 4.2),
        _make_obstacle([28.0, -6.0, -30.0], 5.0),
        _make_obstacle([42.0, 10.0, -12.0], 4.8),
        _make_obstacle([58.0, 28.0, 2.0], 5.4),
        _make_obstacle([68.0, 44.0, 12.0], 4.6),
    ]
    dyn_obs = [
        _make_obstacle([24.0, -28.0, -38.0], 3.6, [0.35, 0.55, 0.22]),
        _make_obstacle([46.0, 8.0, -8.0], 3.9, [-0.40, 0.32, 0.18]),
        _make_obstacle([62.0, 34.0, 8.0], 3.5, [0.28, -0.36, -0.14]),
    ]

    # traj_global = planGlobalBezierPSO(fs, fg, opts, static_obs + dyn_obs)
    traj_global = _build_reference_path_from_tracking_template(fs["p"], fg["p"])
    s_max = traj_global["xyz"].shape[0]
    s_grid = np.arange(1, s_max + 1, dtype=np.float64)
    fx = lambda s: np.interp(s, s_grid, traj_global["xyz"][:, 0])
    fy = lambda s: np.interp(s, s_grid, traj_global["xyz"][:, 1])
    fz = lambda s: np.interp(s, s_grid, traj_global["xyz"][:, 2])

    sensor_params = {
        "range": 12.0,
        "fov_angle": 45.0,
        "danger_distance": 6.0,
        "dt": cfg.dt,
        "num_rays": 100,
    }

    fish_state = np.zeros(13, dtype=np.float64)
    initial_tangent = _path_tangent(traj_global["xyz"], 0)
    initial_tangent = initial_tangent / max(np.linalg.norm(initial_tangent), 1e-8)
    fish_state[:3] = np.array([max(cruise_speed, 0.6), 0.0, 0.0], dtype=np.float64)
    fish_state[Q0 : Q3 + 1] = _quat_from_forward_vector(initial_tangent)
    fish_state[[PX, PY, PZ]] = fs["p"]
    body_params = load_body_params()
    fin_params = load_fin_params()

    ctrl_state = {
        "e_theta_prev": 0.0,
        "e_theta_int": 0.0,
        "e_psi_prev": 0.0,
        "e_psi_int": 0.0,
    }
    hist = np.zeros(5, dtype=np.float64)
    c_A = 5.0
    fin_f = 4.0
    ref_min = np.asarray(env_cfg["ref_min"], dtype=np.float64)
    ref_max = np.asarray(env_cfg["ref_max"], dtype=np.float64)
    fish_params = {
        key: float(value) for key, value in env_cfg["controller_params"].items()
    }
    if controller_params_override:
        for key, value in controller_params_override.items():
            fish_params[key] = float(value)

    local_planner = RLLocalPlanner(model_path=model_path)
    curr_pos = fs["p"].copy()
    robot_vel = fs["v"].copy()
    mode = "GLOBAL_TRACKING"
    current_path_idx = 1
    path_hist = [curr_pos.copy()]
    dyn_obs_hist = [np.asarray([obs["c"].copy() for obs in dyn_obs], dtype=np.float64)]
    blocked_pts_hist: list[np.ndarray] = [np.zeros((0, 3), dtype=np.float64)]
    los_target_hist = [np.full(3, np.nan, dtype=np.float64)]
    local_target_hist = [np.full(3, np.nan, dtype=np.float64)]
    title_hist = ["Starting Simulation with LOS Guidance..."]
    danger_dist_hist = [np.nan]
    reached_goal = False
    numerical_issue = False
    goal_threshold = cfg.goal_threshold

    sa_settings["cmd_vel_des"] = np.zeros((3, cfg.max_steps), dtype=np.float64)
    sa_settings["vel_act"] = np.zeros((3, cfg.max_steps), dtype=np.float64)
    sa_settings["theta_ref"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["theta_act"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["e_theta"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["theta_int"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["psi_ref"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["psi_act"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["e_psi"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["psi_int"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["alpha5_ref"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["delta_ref"] = np.full(cfg.max_steps, np.nan, dtype=np.float64)
    sa_settings["mode"] = []

    for k in range(1, cfg.max_steps + 1):
        for obstacle in dyn_obs:
            obstacle["c"] = obstacle["c"] + obstacle["v"] * cfg.dt
            if obstacle["c"][0] < scene_min[0] or obstacle["c"][0] > scene_max[0]:
                obstacle["v"][0] = -obstacle["v"][0]
                obstacle["c"][0] = np.clip(obstacle["c"][0], scene_min[0], scene_max[0])
            if obstacle["c"][1] < scene_min[1] or obstacle["c"][1] > scene_max[1]:
                obstacle["v"][1] = -obstacle["v"][1]
                obstacle["c"][1] = np.clip(obstacle["c"][1], scene_min[1], scene_max[1])
            if obstacle["c"][2] < scene_min[2] or obstacle["c"][2] > scene_max[2]:
                obstacle["v"][2] = -obstacle["v"][2]
                obstacle["c"][2] = np.clip(obstacle["c"][2], scene_min[2], scene_max[2])

        all_true_obs = static_obs + dyn_obs
        safe_zone = get_visible_obstacles(curr_pos, robot_vel, all_true_obs, sensor_params)
        end_pts = safe_zone["origin"].reshape(3, 1) + safe_zone["rays"] * safe_zone["dists"].reshape(1, -1)
        blocked_pts = end_pts[:, safe_zone["is_blocked"]].T.copy()

        visible_obs = []
        for obstacle in all_true_obs:
            for ray_idx in range(safe_zone["rays"].shape[1]):
                if not safe_zone["is_blocked"][ray_idx]:
                    continue
                hit_pt = safe_zone["origin"] + safe_zone["rays"][:, ray_idx] * safe_zone["dists"][ray_idx]
                if abs(np.linalg.norm(hit_pt - obstacle["c"]) - obstacle["r"]) < 0.1:
                    visible_obs.append(obstacle)
                    break

        d_min = np.inf
        if visible_obs:
            for obstacle in visible_obs:
                d_min = min(d_min, np.linalg.norm(curr_pos - obstacle["c"]) - obstacle["r"])
        elif np.any(safe_zone["is_blocked"]):
            d_min = float(np.min(safe_zone["dists"][safe_zone["is_blocked"]]))

        is_obs_in_danger = np.isfinite(d_min) and d_min < sensor_params["danger_distance"]
        # if mode == "GLOBAL_TRACKING" and is_obs_in_danger:
        #     mode = "LOCAL_AVOIDANCE"
        #     local_planner.reset()
        # elif mode == "LOCAL_AVOIDANCE" and not is_obs_in_danger:
        #     mode = "GLOBAL_TRACKING"

        sa_settings["mode"].append(mode)
        t_k = (k - 1) * cfg.dt
        los_target = np.full(3, np.nan, dtype=np.float64)
        local_target_log = np.full(3, np.nan, dtype=np.float64)

        if mode == "LOCAL_AVOIDANCE":
            local_target = find_local_target(traj_global, curr_pos, cfg.local_lookahead)
            local_target_log = np.asarray(local_target, dtype=np.float64).copy()
            action_ref, _, _, cmd_vel_global, ctrl_state = local_planner.plan(
                fish_state,
                hist,
                local_target,
                visible_obs,
                ctrl_state,
                cfg.dt,
                fish_params,
            )
            sa_settings["cmd_vel_des"][:, k - 1] = cmd_vel_global
            if np.isfinite(d_min):
                title_hist.append(f"Step {k}: LOCAL AVOIDANCE (RL) - Visible Dist: {d_min:.2f}")
            else:
                title_hist.append(f"Step {k}: LOCAL AVOIDANCE (RL)")
        else:
            lookback = max(1, current_path_idx - 10)
            _, _, psi_ref, theta_ref = los_guidance_3d(
                curr_pos, (lookback, s_max), los_params["Delta"], los_params["Delta"], fx, fy, fz
            )
            dists = np.sqrt(np.sum((traj_global["xyz"] - curr_pos.reshape(1, 3)) ** 2, axis=1))
            proj_idx = int(np.argmin(dists))
            current_path_idx = max(current_path_idx, proj_idx + 1)
            nearest_p = traj_global["xyz"][proj_idx]

            cmd_vel_global = np.array(
                [
                    cruise_speed * np.cos(theta_ref) * np.cos(psi_ref),
                    cruise_speed * np.cos(theta_ref) * np.sin(psi_ref),
                    -cruise_speed * np.sin(theta_ref),
                ],
                dtype=np.float64,
            )
            sa_settings["cmd_vel_des"][:, k - 1] = cmd_vel_global
            a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, ctrl_state = fin_controller(
                cmd_vel_global, fish_state, ctrl_state, cfg.dt, fish_params
            )
            action_ref = np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64)
            los_target = np.array(
                [
                    nearest_p[0] + los_params["Delta"] * np.cos(theta_ref) * np.cos(psi_ref),
                    nearest_p[1] + los_params["Delta"] * np.cos(theta_ref) * np.sin(psi_ref),
                    nearest_p[2] - los_params["Delta"] * np.sin(theta_ref),
                ],
                dtype=np.float64,
            )
            title_hist.append(
                f"Step {k}: GLOBAL TRACKING (LOS) - {'No obstacles in FOV' if not np.isfinite(d_min) else f'Visible Dist: {d_min:.2f}'}"
            )

        sa_settings["theta_ref"][k - 1] = float(ctrl_state.get("theta_ref", np.nan))
        sa_settings["theta_act"][k - 1] = float(ctrl_state.get("theta", np.nan))
        sa_settings["e_theta"][k - 1] = float(ctrl_state.get("e_theta", np.nan))
        sa_settings["theta_int"][k - 1] = float(ctrl_state.get("e_theta_int", np.nan))
        sa_settings["psi_ref"][k - 1] = float(ctrl_state.get("psi_ref", np.nan))
        sa_settings["psi_act"][k - 1] = float(ctrl_state.get("psi", np.nan))
        sa_settings["e_psi"][k - 1] = float(ctrl_state.get("e_psi", np.nan))
        sa_settings["psi_int"][k - 1] = float(ctrl_state.get("e_psi_int", np.nan))
        sa_settings["alpha5_ref"][k - 1] = float(ctrl_state.get("alpha5_ref", np.nan))
        sa_settings["delta_ref"][k - 1] = float(ctrl_state.get("delta_ref", np.nan))

        action_ref = np.clip(action_ref, ref_min, ref_max)
        fish_state, hist = dynamics_step(
            fish_state,
            body_params,
            fin_params,
            cfg.dt,
            t_k,
            action_ref,
            hist,
            c_A,
            fin_f,
        )
        if not np.isfinite(fish_state).all():
            numerical_issue = True
            blocked_pts_hist.append(blocked_pts)
            los_target_hist.append(los_target)
            local_target_hist.append(local_target_log)
            danger_dist_hist.append(d_min if np.isfinite(d_min) else np.nan)
            break

        curr_pos = fish_state[[PX, PY, PZ]].copy()
        q0, q1, q2, q3 = fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
        rie = np.array(
            [
                [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
                [2 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q0 * q1 + q3 * q2)],
                [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
            ],
            dtype=np.float64,
        )
        robot_vel = rie.T @ fish_state[:3]
        sa_settings["vel_act"][:, k - 1] = robot_vel
        path_hist.append(curr_pos.copy())
        dyn_obs_hist.append(np.asarray([obs["c"].copy() for obs in dyn_obs], dtype=np.float64))
        blocked_pts_hist.append(blocked_pts)
        los_target_hist.append(los_target)
        local_target_hist.append(local_target_log)
        danger_dist_hist.append(d_min if np.isfinite(d_min) else np.nan)

        if np.linalg.norm(curr_pos - fg["p"]) < goal_threshold:
            reached_goal = True
            break

    steps_executed = len(path_hist) - 1
    result = {
        "path_hist": np.asarray(path_hist, dtype=np.float64),
        "traj_global": traj_global,
        "dyn_obs_hist": np.asarray(dyn_obs_hist, dtype=np.float64),
        "dyn_obs_radii": np.asarray([obs["r"] for obs in dyn_obs], dtype=np.float64),
        "blocked_pts_hist": blocked_pts_hist,
        "los_target_hist": np.asarray(los_target_hist, dtype=np.float64),
        "local_target_hist": np.asarray(local_target_hist, dtype=np.float64),
        "title_hist": title_hist,
        "danger_dist_hist": np.asarray(danger_dist_hist, dtype=np.float64),
        "settings": sa_settings,
        "steps_executed": steps_executed,
        "sensor_params": sensor_params,
        "initial_robot_vel": fs["v"].copy(),
        "reached_goal": reached_goal,
        "numerical_issue": numerical_issue,
        "goal": fg["p"].copy(),
        "final_state": fish_state,
        "final_hist": hist,
    }
    result["tracking_metrics"] = _compute_tracking_metrics(sa_settings, fish_params, steps_executed)
    if visualize:
        from simulation.visualization import plot_hybrid_result

        plot_hybrid_result(result, static_obs, dyn_obs, fg["p"])
    return result


def main() -> None:
    result = run_hybrid_los_rl_simulation(visualize=True)
    print("reached_goal:", result["reached_goal"])
    print("path_len:", len(result["path_hist"]))
    print("tracking_metrics:", result["tracking_metrics"])


if __name__ == "__main__":
    main()
