from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running this file directly from the repo root or an IDE run config.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dynamics import step as dynamics_step
from dynamics.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, VX, VY, VZ
from dynamics.params import load_body_params, load_fin_params
from rl_training.config.fish_env_config import build_fish_env_config
from simulation.fin_controller.controller import fin_controller
from simulation.global_planner.los import los_guidance_3d
from simulation.main.visualization import _axis_limits_from_points, _set_3d_axis_scale


@dataclass
class LOSTrackingTestConfig:
    passive_mode: bool = False
    dt: float = 0.2
    max_steps: int = 500
    goal_threshold: float = 1.0
    initial_speed: float = 0.0
    cruise_speed: float = 0.12
    speed_ramp_time: float = 0.0
    los_delta: float = 2.5
    lookback_margin: int = 20
    samples_per_segment: int = 60
    c_A: float = 5.0
    fin_f: float = 4.0
    rot_base_override_deg: float | None = None


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


def _rotation_world_from_quat(quat: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = [float(v) for v in quat]
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2.0 * (q1 * q2 + q0 * q3), 2.0 * (q1 * q3 - q0 * q2)],
            [2.0 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2.0 * (q0 * q1 + q3 * q2)],
            [2.0 * (q1 * q3 + q0 * q2), 2.0 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=np.float64,
    )


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


def build_complex_tracking_path(samples_per_segment: int = 60) -> dict:
    # Keep x monotonic while varying lateral and depth motion, so the path is richer
    # than a straight line but still physically reasonable for the current controller.
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [6.0, 1.0, -0.8],
            [12.0, -2.2, -0.3],
            [18.0, 2.4, -1.9],
            [24.0, 3.6, 0.5],
            [30.0, -2.8, 1.0],
            [36.0, 1.8, -1.1],
            [42.0, 4.3, 2.0],
            [48.0, 3.7, 4.2],
            [54.0, 1.4, 1.4],
            [60.0, 0.0, 0.0]
        ],
        dtype=np.float64,
    )
    xyz = _catmull_rom_chain(waypoints, samples_per_segment)
    return {"waypoints": waypoints, "xyz": xyz}


def _compute_metrics(result: dict, controller_params: dict) -> dict:
    cross_track_err = np.asarray(result["cross_track_err_hist"], dtype=np.float64)
    theta_err = np.asarray(result["theta_err_hist"], dtype=np.float64)
    psi_err = np.asarray(result["psi_err_hist"], dtype=np.float64)
    alpha5_ref = np.asarray(result["alpha5_ref_hist"], dtype=np.float64)
    delta_ref = np.asarray(result["delta_ref_hist"], dtype=np.float64)

    alpha5_limit = max(abs(float(controller_params["alpha5_min"])), abs(float(controller_params["alpha5_max"])))
    delta_limit = abs(float(controller_params["delta_rot_max"]))
    sat_tol = np.deg2rad(0.5)

    return {
        "goal_distance": float(np.linalg.norm(result["path_hist"][-1] - result["goal"])),
        "cross_track_mean": float(np.mean(cross_track_err)),
        "cross_track_rmse": float(np.sqrt(np.mean(cross_track_err**2))),
        "cross_track_max": float(np.max(cross_track_err)),
        "theta_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(theta_err**2)))),
        "psi_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(psi_err**2)))),
        "alpha5_sat_pct": float(100.0 * np.mean(np.abs(np.abs(alpha5_ref) - alpha5_limit) <= sat_tol)),
        "delta_sat_pct": float(100.0 * np.mean(np.abs(np.abs(delta_ref) - delta_limit) <= sat_tol)),
    }


def _build_controller_params(
    env_cfg: dict,
    cfg: LOSTrackingTestConfig,
    controller_params_override: dict | None = None,
) -> dict:
    fish_params = {key: float(value) for key, value in env_cfg["controller_params"].items()}
    if cfg.rot_base_override_deg is not None:
        fish_params["A_rot_base"] = float(np.deg2rad(cfg.rot_base_override_deg))
    if controller_params_override:
        for key, value in controller_params_override.items():
            fish_params[key] = float(value)
    return fish_params


def _command_speed(cfg: LOSTrackingTestConfig, t_now: float) -> float:
    if cfg.speed_ramp_time <= 1e-9:
        return float(cfg.cruise_speed)
    tau = float(np.clip(t_now / cfg.speed_ramp_time, 0.0, 1.0))
    tau = tau * tau * (3.0 - 2.0 * tau)
    return float(cfg.cruise_speed * tau)


def run_passive_drift_test(
    visualize: bool = True,
    save_path: str | None = None,
    config: LOSTrackingTestConfig | None = None,
    animate: bool = True,
) -> dict:
    cfg = config or LOSTrackingTestConfig()
    env_cfg = build_fish_env_config()
    body_params = load_body_params()
    fin_params = load_fin_params()
    traj = build_complex_tracking_path(cfg.samples_per_segment)
    traj_xyz = traj["xyz"]
    s_max = traj_xyz.shape[0]
    s_grid = np.arange(1, s_max + 1, dtype=np.float64)

    fx = lambda s: np.interp(s, s_grid, traj_xyz[:, 0])
    fy = lambda s: np.interp(s, s_grid, traj_xyz[:, 1])
    fz = lambda s: np.interp(s, s_grid, traj_xyz[:, 2])

    fish_params = _build_controller_params(env_cfg, cfg)
    ref_min = np.asarray(env_cfg["ref_min"], dtype=np.float64)
    ref_max = np.asarray(env_cfg["ref_max"], dtype=np.float64)

    fish_state = np.zeros(13, dtype=np.float64)
    fish_state[Q0 : Q3 + 1] = _quat_from_forward_vector(_path_tangent(traj_xyz, 0))
    fish_state[[PX, PY, PZ]] = traj_xyz[0]
    initial_rot_world = _rotation_world_from_quat(fish_state[Q0 : Q3 + 1])
    initial_vel_world = initial_rot_world.T @ fish_state[:3]
    hist = np.zeros(5, dtype=np.float64)
    ctrl_state = {
        "e_theta_prev": 0.0,
        "e_theta_int": 0.0,
        "e_psi_prev": 0.0,
        "e_psi_int": 0.0,
    }
    current_path_idx = 1
    initial_tangent = _path_tangent(traj_xyz, 0)
    initial_tangent = initial_tangent / max(np.linalg.norm(initial_tangent), 1e-8)

    path_hist = [fish_state[[PX, PY, PZ]].copy()]
    proj_hist = [traj_xyz[0].copy()]
    los_target_hist = [traj_xyz[0].copy()]
    cmd_vel_hist = [np.zeros(3, dtype=np.float64)]
    act_vel_hist = [initial_vel_world.copy()]
    cross_track_err_hist = [0.0]
    theta_err_hist = [0.0]
    psi_err_hist = [0.0]
    alpha5_ref_hist = [0.0]
    delta_ref_hist = [0.0]
    reached_goal = False
    numerical_issue = False

    for k in range(1, cfg.max_steps + 1):
        curr_pos = fish_state[[PX, PY, PZ]].copy()
        dists = np.linalg.norm(traj_xyz - curr_pos.reshape(1, 3), axis=1)
        proj_idx = int(np.argmin(dists))
        current_path_idx = max(current_path_idx, proj_idx + 1)
        nearest_p = traj_xyz[proj_idx]

        lookback = max(1, current_path_idx - cfg.lookback_margin)
        _, _, psi_ref, theta_ref = los_guidance_3d(
            curr_pos,
            (lookback, s_max),
            cfg.los_delta,
            cfg.los_delta,
            fx,
            fy,
            fz,
        )
        cmd_speed = _command_speed(cfg, (k - 1) * cfg.dt)
        cmd_vel_global = np.array(
            [
                cmd_speed * np.cos(theta_ref) * np.cos(psi_ref),
                cmd_speed * np.cos(theta_ref) * np.sin(psi_ref),
                -cmd_speed * np.sin(theta_ref),
            ],
            dtype=np.float64,
        )
        a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, ctrl_state = fin_controller(
            psi_ref, theta_ref, cmd_speed, fish_state, ctrl_state, cfg.dt, fish_params
        )
        action_ref = np.clip(
            np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64),
            ref_min,
            ref_max,
        )
        los_target = np.array(
            [
                nearest_p[0] + cfg.los_delta * np.cos(theta_ref) * np.cos(psi_ref),
                nearest_p[1] + cfg.los_delta * np.cos(theta_ref) * np.sin(psi_ref),
                nearest_p[2] - cfg.los_delta * np.sin(theta_ref),
            ],
            dtype=np.float64,
        )

        fish_state, hist = dynamics_step(
            fish_state,
            body_params,
            fin_params,
            cfg.dt,
            (k - 1) * cfg.dt,
            action_ref,
            hist,
            cfg.c_A,
            cfg.fin_f,
        )
        if not np.isfinite(fish_state).all():
            numerical_issue = True
            break

        curr_pos = fish_state[[PX, PY, PZ]].copy()
        rot_world = _rotation_world_from_quat(fish_state[Q0 : Q3 + 1])
        vel_world = rot_world.T @ fish_state[:3]
        curr_dists = np.linalg.norm(traj_xyz - curr_pos.reshape(1, 3), axis=1)
        curr_proj = traj_xyz[int(np.argmin(curr_dists))]

        path_hist.append(curr_pos.copy())
        proj_hist.append(curr_proj.copy())
        los_target_hist.append(los_target.copy())
        cmd_vel_hist.append(cmd_vel_global.copy())
        act_vel_hist.append(vel_world.copy())
        cross_track_err_hist.append(float(np.min(curr_dists)))
        theta_err_hist.append(float(ctrl_state.get("e_theta", 0.0)))
        psi_err_hist.append(float(ctrl_state.get("e_psi", 0.0)))
        alpha5_ref_hist.append(float(ctrl_state.get("alpha5_ref", 0.0)))
        delta_ref_hist.append(float(ctrl_state.get("delta_ref", 0.0)))

        if np.linalg.norm(curr_pos - traj_xyz[-1]) < cfg.goal_threshold:
            reached_goal = True
            break

    result = {
        "config": cfg,
        "traj": traj,
        "goal": traj_xyz[-1].copy(),
        "path_hist": np.asarray(path_hist, dtype=np.float64),
        "proj_hist": np.asarray(proj_hist, dtype=np.float64),
        "los_target_hist": np.asarray(los_target_hist, dtype=np.float64),
        "cmd_vel_hist": np.asarray(cmd_vel_hist, dtype=np.float64),
        "act_vel_hist": np.asarray(act_vel_hist, dtype=np.float64),
        "cross_track_err_hist": np.asarray(cross_track_err_hist, dtype=np.float64),
        "theta_err_hist": np.asarray(theta_err_hist, dtype=np.float64),
        "psi_err_hist": np.asarray(psi_err_hist, dtype=np.float64),
        "alpha5_ref_hist": np.asarray(alpha5_ref_hist, dtype=np.float64),
        "delta_ref_hist": np.asarray(delta_ref_hist, dtype=np.float64),
        "reached_goal": reached_goal,
        "numerical_issue": numerical_issue,
        "steps_executed": len(path_hist) - 1,
        "initial_tangent": initial_tangent.copy(),
    }
    result["metrics"] = _compute_metrics(result, fish_params)

    if visualize or save_path:
        _plot_los_fin_tracking_result(result, save_path=save_path, show=visualize, animate=animate)

    return result


def run_los_fin_tracking_test(
    visualize: bool = True,
    save_path: str | None = None,
    controller_params_override: dict | None = None,
    config: LOSTrackingTestConfig | None = None,
    animate: bool = True,
) -> dict:
    cfg = config or LOSTrackingTestConfig()
    env_cfg = build_fish_env_config()
    body_params = load_body_params()
    fin_params = load_fin_params()
    traj = build_complex_tracking_path(cfg.samples_per_segment)
    traj_xyz = traj["xyz"]
    s_max = traj_xyz.shape[0]
    s_grid = np.arange(1, s_max + 1, dtype=np.float64)

    fx = lambda s: np.interp(s, s_grid, traj_xyz[:, 0])
    fy = lambda s: np.interp(s, s_grid, traj_xyz[:, 1])
    fz = lambda s: np.interp(s, s_grid, traj_xyz[:, 2])

    fish_params = _build_controller_params(env_cfg, cfg, controller_params_override)

    ref_min = np.asarray(env_cfg["ref_min"], dtype=np.float64)
    ref_max = np.asarray(env_cfg["ref_max"], dtype=np.float64)

    fish_state = np.zeros(13, dtype=np.float64)
    fish_state[Q0 : Q3 + 1] = _quat_from_forward_vector(_path_tangent(traj_xyz, 0))
    fish_state[[VX, VY, VZ]] = [cfg.initial_speed, 0.0, 0.0]
    fish_state[[PX, PY, PZ]] = traj_xyz[0]
    initial_rot_world = _rotation_world_from_quat(fish_state[Q0 : Q3 + 1])
    initial_vel_world = initial_rot_world.T @ fish_state[:3]
    initial_tangent = _path_tangent(traj_xyz, 0)
    initial_tangent = initial_tangent / max(np.linalg.norm(initial_tangent), 1e-8)

    ctrl_state = {
        "e_theta_prev": 0.0,
        "e_theta_int": 0.0,
        "e_psi_prev": 0.0,
        "e_psi_int": 0.0,
    }
    hist = np.zeros(5, dtype=np.float64)
    current_path_idx = 1

    path_hist = [fish_state[[PX, PY, PZ]].copy()]
    proj_hist = [traj_xyz[0].copy()]
    los_target_hist = [traj_xyz[0].copy()]
    cmd_vel_hist: list[np.ndarray] = []
    act_vel_hist = [initial_vel_world.copy()]
    cross_track_err_hist: list[float] = [0.0]
    theta_err_hist: list[float] = [0.0]
    psi_err_hist: list[float] = [0.0]
    alpha5_ref_hist: list[float] = [0.0]
    delta_ref_hist: list[float] = [0.0]
    reached_goal = False
    numerical_issue = False

    for k in range(1, cfg.max_steps + 1):
        curr_pos = fish_state[[PX, PY, PZ]].copy()
        dists = np.linalg.norm(traj_xyz - curr_pos.reshape(1, 3), axis=1)
        proj_idx = int(np.argmin(dists))
        current_path_idx = max(current_path_idx, proj_idx + 1)
        nearest_p = traj_xyz[proj_idx]

        lookback = max(1, current_path_idx - cfg.lookback_margin)
        _, _, psi_ref, theta_ref = los_guidance_3d(
            curr_pos,
            (lookback, s_max),
            cfg.los_delta,
            cfg.los_delta,
            fx,
            fy,
            fz,
        )
        cmd_speed = _command_speed(cfg, (k - 1) * cfg.dt)
        cmd_vel_global = np.array(
            [
                cmd_speed * np.cos(theta_ref) * np.cos(psi_ref),
                cmd_speed * np.cos(theta_ref) * np.sin(psi_ref),
                -cmd_speed * np.sin(theta_ref),
            ],
            dtype=np.float64,
        )
        a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, ctrl_state = fin_controller(
            psi_ref, theta_ref, cmd_speed, fish_state, ctrl_state, cfg.dt, fish_params
        )
        action_ref = np.clip(np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64), ref_min, ref_max)
        los_target = np.array(
            [
                nearest_p[0] + cfg.los_delta * np.cos(theta_ref) * np.cos(psi_ref),
                nearest_p[1] + cfg.los_delta * np.cos(theta_ref) * np.sin(psi_ref),
                nearest_p[2] - cfg.los_delta * np.sin(theta_ref),
            ],
            dtype=np.float64,
        )

        fish_state, hist = dynamics_step(
            fish_state,
            body_params,
            fin_params,
            cfg.dt,
            (k - 1) * cfg.dt,
            action_ref,
            hist,
            cfg.c_A,
            cfg.fin_f,
        )
        if not np.isfinite(fish_state).all():
            numerical_issue = True
            break

        curr_pos = fish_state[[PX, PY, PZ]].copy()
        rot_world = _rotation_world_from_quat(fish_state[Q0 : Q3 + 1])
        vel_world = rot_world.T @ fish_state[:3]

        curr_dists = np.linalg.norm(traj_xyz - curr_pos.reshape(1, 3), axis=1)
        curr_proj_idx = int(np.argmin(curr_dists))
        curr_proj = traj_xyz[curr_proj_idx]

        path_hist.append(curr_pos.copy())
        proj_hist.append(curr_proj.copy())
        los_target_hist.append(los_target.copy())
        cmd_vel_hist.append(cmd_vel_global.copy())
        act_vel_hist.append(vel_world.copy())
        cross_track_err_hist.append(float(curr_dists[curr_proj_idx]))
        theta_err_hist.append(float(ctrl_state.get("e_theta", 0.0)))
        psi_err_hist.append(float(ctrl_state.get("e_psi", 0.0)))
        alpha5_ref_hist.append(float(ctrl_state.get("alpha5_ref", 0.0)))
        delta_ref_hist.append(float(ctrl_state.get("delta_ref", 0.0)))

        if np.linalg.norm(curr_pos - traj_xyz[-1]) < cfg.goal_threshold:
            reached_goal = True
            break

    result = {
        "config": cfg,
        "traj": traj,
        "goal": traj_xyz[-1].copy(),
        "path_hist": np.asarray(path_hist, dtype=np.float64),
        "proj_hist": np.asarray(proj_hist, dtype=np.float64),
        "los_target_hist": np.asarray(los_target_hist, dtype=np.float64),
        "cmd_vel_hist": np.asarray(cmd_vel_hist, dtype=np.float64),
        "act_vel_hist": np.asarray(act_vel_hist, dtype=np.float64),
        "cross_track_err_hist": np.asarray(cross_track_err_hist, dtype=np.float64),
        "theta_err_hist": np.asarray(theta_err_hist, dtype=np.float64),
        "psi_err_hist": np.asarray(psi_err_hist, dtype=np.float64),
        "alpha5_ref_hist": np.asarray(alpha5_ref_hist, dtype=np.float64),
        "delta_ref_hist": np.asarray(delta_ref_hist, dtype=np.float64),
        "reached_goal": reached_goal,
        "numerical_issue": numerical_issue,
        "steps_executed": len(path_hist) - 1,
        "initial_tangent": initial_tangent.copy(),
    }
    result["metrics"] = _compute_metrics(result, fish_params)

    if visualize or save_path:
        _plot_los_fin_tracking_result(result, save_path=save_path, show=visualize, animate=animate)

    return result


def _plot_los_fin_tracking_result(result: dict, save_path: str | None, show: bool, animate: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    traj_xyz = result["traj"]["xyz"]
    waypoints = result["traj"]["waypoints"]
    path_hist = result["path_hist"]
    proj_hist = result["proj_hist"]
    los_target_hist = result["los_target_hist"]
    initial_tangent = np.asarray(result["initial_tangent"], dtype=np.float64)
    time_axis = np.arange(path_hist.shape[0], dtype=np.float64) * result["config"].dt
    animate = bool(animate and show and path_hist.shape[0] > 1)

    fig = plt.figure(figsize=(15, 10))
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2], "--", color="tab:blue", linewidth=2.0, label="Reference")
    ax3d.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="tab:green", s=26, label="Waypoints")
    if animate:
        tracked_plot, = ax3d.plot(
            path_hist[:1, 0],
            path_hist[:1, 1],
            path_hist[:1, 2],
            color="tab:orange",
            linewidth=2.2,
            label="Tracked",
        )
        robot_plot = ax3d.scatter(
            [path_hist[0, 0]],
            [path_hist[0, 1]],
            [path_hist[0, 2]],
            color="tab:orange",
            s=45,
            edgecolors="k",
            zorder=5,
        )
        proj_plot = ax3d.scatter(
            [proj_hist[0, 0]],
            [proj_hist[0, 1]],
            [proj_hist[0, 2]],
            color="tab:purple",
            s=26,
            alpha=0.8,
            label="Projection",
        )
        los_plot = ax3d.scatter(
            [los_target_hist[0, 0]],
            [los_target_hist[0, 1]],
            [los_target_hist[0, 2]],
            color="tab:red",
            s=28,
            alpha=0.85,
            label="LOS Target",
        )
    else:
        tracked_plot, = ax3d.plot(
            path_hist[:, 0],
            path_hist[:, 1],
            path_hist[:, 2],
            color="tab:orange",
            linewidth=2.0,
            label="Tracked",
        )
        robot_plot = ax3d.scatter(
            [path_hist[-1, 0]],
            [path_hist[-1, 1]],
            [path_hist[-1, 2]],
            color="tab:orange",
            s=45,
            edgecolors="k",
            zorder=5,
        )
        proj_plot = ax3d.scatter(
            [proj_hist[-1, 0]],
            [proj_hist[-1, 1]],
            [proj_hist[-1, 2]],
            color="tab:purple",
            s=26,
            alpha=0.8,
            label="Projection",
        )
        stride = max(1, path_hist.shape[0] // 30)
        los_plot = ax3d.scatter(
            los_target_hist[::stride, 0],
            los_target_hist[::stride, 1],
            los_target_hist[::stride, 2],
            color="tab:red",
            s=14,
            alpha=0.55,
            label="LOS Target",
        )
    ax3d.scatter(
        [path_hist[0, 0]],
        [path_hist[0, 1]],
        [path_hist[0, 2]],
        color="tab:cyan",
        s=36,
        edgecolors="k",
        label="Start",
    )
    tangent_len = max(1.0, 0.08 * np.linalg.norm(traj_xyz[-1] - traj_xyz[0]))
    tangent_plot, = ax3d.plot(
        [
            path_hist[0, 0],
            path_hist[0, 0] + tangent_len * initial_tangent[0],
        ],
        [
            path_hist[0, 1],
            path_hist[0, 1] + tangent_len * initial_tangent[1],
        ],
        [
            path_hist[0, 2],
            path_hist[0, 2] + tangent_len * initial_tangent[2],
        ],
        color="tab:cyan",
        linewidth=2.0,
        alpha=0.9,
        label="Initial Tangent",
    )
    ax3d.set_title("LOS + Fin Controller 3D Tracking")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    axis_mins, axis_maxs = _axis_limits_from_points(
        [
            np.asarray(traj_xyz, dtype=np.float64),
            np.asarray(waypoints, dtype=np.float64),
            np.asarray(path_hist, dtype=np.float64),
            np.asarray(proj_hist, dtype=np.float64),
            np.asarray(los_target_hist, dtype=np.float64),
        ],
        min_span=2.0,
        pad_ratio=0.08,
    )
    _set_3d_axis_scale(ax3d, axis_mins, axis_maxs)
    ax3d.view_init(elev=28.0, azim=-55.0)
    ax3d.legend(loc="best")

    ax_err = fig.add_subplot(222)
    ax_err.plot(time_axis, result["cross_track_err_hist"], color="tab:blue", linewidth=1.8, label="Cross-track error")
    ax_err.plot(time_axis, np.linalg.norm(path_hist - proj_hist, axis=1), color="tab:purple", linewidth=1.2, alpha=0.75, label="Proj distance")
    ax_err.set_title("Tracking Error")
    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    ax_heading = fig.add_subplot(223)
    ax_heading.plot(time_axis, np.rad2deg(result["psi_err_hist"]), color="tab:orange", linewidth=1.6, label="Yaw error")
    ax_heading.plot(time_axis, np.rad2deg(result["theta_err_hist"]), color="tab:green", linewidth=1.6, label="Pitch error")
    ax_heading.set_title("Attitude Tracking Error")
    ax_heading.set_xlabel("Time [s]")
    ax_heading.set_ylabel("Error [deg]")
    ax_heading.grid(True, alpha=0.3)
    ax_heading.legend(loc="best")

    ax_ctrl = fig.add_subplot(224)
    ax_ctrl.plot(time_axis, np.rad2deg(result["delta_ref_hist"]), color="tab:red", linewidth=1.6, label="delta_ref")
    ax_ctrl.plot(time_axis, np.rad2deg(result["alpha5_ref_hist"]), color="tab:brown", linewidth=1.6, label="alpha5_ref")
    ax_ctrl.set_title("Controller Output")
    ax_ctrl.set_xlabel("Time [s]")
    ax_ctrl.set_ylabel("Command [deg]")
    ax_ctrl.grid(True, alpha=0.3)
    ax_ctrl.legend(loc="best")

    metrics = result["metrics"]
    if "goal_distance" in metrics:
        title_suffix = (
            f"Reached Goal: {result['reached_goal']} | "
            f"Goal Dist: {metrics['goal_distance']:.2f} m | "
            f"CTE RMSE: {metrics['cross_track_rmse']:.2f} m"
        )
    else:
        title_suffix = (
            f"Passive Drift | "
            f"Final Drift: {metrics['final_distance_from_start']:.4f} m | "
            f"Max Speed: {metrics['max_speed']:.4f} m/s"
        )
    fig.suptitle(
        "LOS/Fin Tracking Test | " + title_suffix
    )
    fig.tight_layout()

    if animate:
        interval_ms = max(15, int(1000.0 * result["config"].dt * 0.2))

        def _update(frame_idx: int):
            tracked_plot.set_data(path_hist[: frame_idx + 1, 0], path_hist[: frame_idx + 1, 1])
            tracked_plot.set_3d_properties(path_hist[: frame_idx + 1, 2])
            robot_plot._offsets3d = (
                [path_hist[frame_idx, 0]],
                [path_hist[frame_idx, 1]],
                [path_hist[frame_idx, 2]],
            )
            proj_plot._offsets3d = (
                [proj_hist[frame_idx, 0]],
                [proj_hist[frame_idx, 1]],
                [proj_hist[frame_idx, 2]],
            )
            los_plot._offsets3d = (
                [los_target_hist[frame_idx, 0]],
                [los_target_hist[frame_idx, 1]],
                [los_target_hist[frame_idx, 2]],
            )
            tangent_plot.set_data(
                [
                    path_hist[0, 0],
                    path_hist[0, 0] + tangent_len * initial_tangent[0],
                ],
                [
                    path_hist[0, 1],
                    path_hist[0, 1] + tangent_len * initial_tangent[1],
                ],
            )
            tangent_plot.set_3d_properties(
                [
                    path_hist[0, 2],
                    path_hist[0, 2] + tangent_len * initial_tangent[2],
                ]
            )
            ax3d.set_title(
                "LOS + Fin Controller 3D Tracking "
                f"| Step {frame_idx:03d} "
                f"| CTE {result['cross_track_err_hist'][frame_idx]:.2f} m"
            )
            return tracked_plot, robot_plot, proj_plot, los_plot, tangent_plot

        fig._tracking_anim = FuncAnimation(
            fig,
            _update,
            frames=path_hist.shape[0],
            interval=interval_ms,
            blit=False,
            repeat=True,
        )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone LOS + fin_controller tracking test.")
    parser.add_argument("--no-plot", action="store_true", help="Run the test without opening a matplotlib window.")
    parser.add_argument("--passive", action="store_true", help="Start from full rest and apply zero fin commands throughout the simulation.")
    parser.add_argument("--static", action="store_true", help="Disable 3D animation and show only the final static plot.")
    parser.add_argument("--save", type=str, default=None, help="Optional figure output path, e.g. outputs/los_fin_test.png")
    parser.add_argument("--speed", type=float, default=0.12, help="Commanded cruise speed used by LOS guidance.")
    parser.add_argument("--los-delta", type=float, default=2.5, help="LOS look-ahead distance.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum simulation steps.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = LOSTrackingTestConfig(
        passive_mode=True,
        cruise_speed=float(args.speed),
        speed_ramp_time=3.0,
        los_delta=float(args.los_delta),
        max_steps=int(args.max_steps),
        rot_base_override_deg=None,
    )
    if cfg.passive_mode or args.passive:
        result = run_passive_drift_test(
            visualize=not args.no_plot,
            save_path=args.save,
            config=cfg,
            animate=not args.static,
        )
    else:
        result = run_los_fin_tracking_test(
            visualize=not args.no_plot,
            save_path=args.save,
            config=cfg,
            animate=not args.static,
        )
    print("reached_goal:", result["reached_goal"])
    print("steps_executed:", result["steps_executed"])
    print("metrics:", result["metrics"])


if __name__ == "__main__":
    main()
