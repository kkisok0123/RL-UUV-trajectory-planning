from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from dynamics import step as dynamics_step
from dynamics.indices import PX, PY, PZ, Q0, Q1, Q2, Q3, VX, VY, VZ
from rl_training.config.fish_env_config import build_fish_env_config
from rl_training.envs.geometry import nearest_obstacle_info
from simulation.fin_controller.controller import fin_controller
from simulation.global_planner.los import los_guidance_3d
from simulation.local_planner.common import find_local_target, get_visible_obstacles
from simulation.local_planner.mpc_planner.planner import MPCLocalPlanner
from simulation.local_planner.rl_planner.planner import RLLocalPlanner


@dataclass
class HybridSimulationConfig:
    """Top-level runtime parameters for the hybrid LOS simulation."""

    dt: float = 0.2
    max_steps: int = 1500
    goal_threshold: float = 1.0
    local_lookahead: float = 7.0
    cruise_speed: float = 0.12
    speed_ramp_time: float = 3.0
    lookback_margin: int = 20


def _make_obstacle(center, radius, velocity=None) -> dict:
    """Build a static obstacle descriptor."""
    if velocity is None:
        velocity = np.zeros(3, dtype=np.float64)
    return {
        "c": np.asarray(center, dtype=np.float64).reshape(3),
        "r": float(radius),
        "v": np.asarray(velocity, dtype=np.float64).reshape(3),
    }


def _make_dynamic_obstacle(
    anchor: np.ndarray,
    radius: float,
    amplitude: np.ndarray,
    frequency: np.ndarray,
    phase: np.ndarray,
) -> dict:
    """Build a harmonic dynamic obstacle descriptor."""
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(3)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(3)
    phase = np.asarray(phase, dtype=np.float64).reshape(3)
    return {
        "c": anchor.copy(),
        "r": float(radius),
        "v": np.zeros(3, dtype=np.float64),
        "motion": "harmonic",
        "anchor": anchor,
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
    }


def _update_dynamic_obstacle(obstacle: dict, t_now: float) -> None:
    """Advance a dynamic obstacle to the requested simulation time."""
    if obstacle.get("motion") != "harmonic":
        obstacle["c"] = obstacle["c"] + obstacle["v"] * 0.0
        return

    anchor = np.asarray(obstacle["anchor"], dtype=np.float64)
    amplitude = np.asarray(obstacle["amplitude"], dtype=np.float64)
    frequency = np.asarray(obstacle["frequency"], dtype=np.float64)
    phase = np.asarray(obstacle["phase"], dtype=np.float64)
    omega_t = frequency * t_now + phase
    offset = np.array(
        [
            amplitude[0] * np.sin(omega_t[0]),
            amplitude[1] * (0.65 * np.sin(omega_t[1]) + 0.35 * np.sin(2.0 * omega_t[1] + 0.4)),
            amplitude[2] * (0.55 * np.cos(omega_t[2]) + 0.45 * np.sin(1.7 * omega_t[2] - 0.3)),
        ],
        dtype=np.float64,
    )
    velocity = np.array(
        [
            amplitude[0] * frequency[0] * np.cos(omega_t[0]),
            amplitude[1]
            * frequency[1]
            * (0.65 * np.cos(omega_t[1]) + 0.70 * np.cos(2.0 * omega_t[1] + 0.4)),
            amplitude[2]
            * frequency[2]
            * (-0.55 * np.sin(omega_t[2]) + 0.765 * np.cos(1.7 * omega_t[2] - 0.3)),
        ],
        dtype=np.float64,
    )
    obstacle["c"] = anchor + offset
    obstacle["v"] = velocity


def _quat_from_forward_vector(forward_world: np.ndarray) -> np.ndarray:
    """Build a quaternion that points the robot along a forward world vector."""
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
    """Interpolate waypoints into a smooth reference path."""
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
    """Estimate the tangent direction along a discretized path."""
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


def _build_reference_path() -> dict:
    """Build the default hand-crafted global path used in hybrid simulations."""
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
            [60.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return {"waypoints": waypoints, "xyz": _catmull_rom_chain(waypoints, samples_per_segment=80)}


def _build_default_obstacles() -> tuple[list[dict], list[dict]]:
    """Build the default static and dynamic obstacles used by hybrid simulations."""
    static_obs = [
        _make_obstacle([16.5, 2.05, -1.73], 2.0),
        _make_obstacle([48.0, 3.7, 4.2], 0.65),
    ]
    dyn_obs = [
        _make_dynamic_obstacle(
            anchor=[31.5, 1.20, 0.55],
            radius=2.5,
            amplitude=[2.0, 4.0, 1.5],
            frequency=[0.11, 0.19, 0.11],
            phase=[0.20, 1.05, -0.35],
        ),
        _make_dynamic_obstacle(
            anchor=[35.0, -1.10, 0.10],
            radius=1.6,
            amplitude=[0.20, 1.20, 1.80],
            frequency=[0.10, 0.15, 0.13],
            phase=[1.10, -0.45, 0.60],
        ),
    ]
    return static_obs, dyn_obs


def _build_sensor_params(cfg: HybridSimulationConfig) -> dict:
    """Build the sensor model used during hybrid simulation."""
    return {
        "range": 8.0,
        "fov_angle": 60.0,
        "danger_distance_enter": 6.0,
        "danger_distance_exit": 4.0,
        "dt": cfg.dt,
        "num_rays": 100,
    }


def _scene_bounds_from_path(
    path_xyz: np.ndarray,
    pad_ratio: float = 0.10,
    min_pad: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute conservative scene bounds around a reference path."""
    path_xyz = np.asarray(path_xyz, dtype=np.float64).reshape(-1, 3)
    spans = np.ptp(path_xyz, axis=0)
    margin = np.maximum(spans * float(pad_ratio), float(min_pad))
    return np.min(path_xyz, axis=0) - margin, np.max(path_xyz, axis=0) + margin


def _build_hybrid_scene(cfg: HybridSimulationConfig) -> dict:
    """Build the reference path, start and goal states, and obstacle scene."""
    traj_global = _build_reference_path()
    path_waypoints = traj_global["waypoints"]
    path_xyz = traj_global["xyz"]
    initial_tangent = _path_tangent(path_xyz, 0)
    initial_tangent = initial_tangent / max(np.linalg.norm(initial_tangent), 1e-8)
    fs = {"p": path_waypoints[0].copy(), "v": cfg.cruise_speed * initial_tangent, "a": np.zeros(3)}
    fg = {"p": path_waypoints[-1].copy(), "v": np.zeros(3), "a": np.zeros(3)}
    scene_min, scene_max = _scene_bounds_from_path(path_xyz)
    static_obs, dyn_obs = _build_default_obstacles()
    return {
        "traj_global": traj_global,
        "fs": fs,
        "fg": fg,
        "scene_min": scene_min,
        "scene_max": scene_max,
        "los_params": {"Delta": 2.5, "k_p": 0.5},
        "static_obs": static_obs,
        "dyn_obs": dyn_obs,
        "sensor_params": _build_sensor_params(cfg),
    }


def _default_ctrl_state() -> dict:
    """Build the default controller integrator state."""
    return {
        "e_theta_prev": 0.0,
        "e_theta_int": 0.0,
        "e_psi_prev": 0.0,
        "e_psi_int": 0.0,
    }


def _command_speed(cfg: HybridSimulationConfig, t_now: float) -> float:
    """Ramp commanded speed smoothly from zero to cruise speed."""
    if cfg.speed_ramp_time <= 1e-9:
        return float(cfg.cruise_speed)
    tau = float(np.clip(t_now / cfg.speed_ramp_time, 0.0, 1.0))
    tau = tau * tau * (3.0 - 2.0 * tau)
    return float(cfg.cruise_speed * tau)


def _compute_tracking_metrics(settings: dict, controller_params: dict, steps_executed: int) -> dict:
    """Summarize controller tracking errors and saturation rates."""
    if steps_executed <= 0:
        return {}
    theta_err = np.asarray(settings.get("e_theta", np.zeros(steps_executed)), dtype=np.float64)[:steps_executed]
    psi_err = np.asarray(settings.get("e_psi", np.zeros(steps_executed)), dtype=np.float64)[:steps_executed]
    alpha5_ref = np.asarray(settings.get("alpha5_ref", np.zeros(steps_executed)), dtype=np.float64)[:steps_executed]
    delta_ref = np.asarray(settings.get("delta_ref", np.zeros(steps_executed)), dtype=np.float64)[:steps_executed]
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


def _build_local_planner_timing(sa_settings: dict, steps_executed: int) -> dict:
    """Build timing diagnostics for local planner calls."""
    local_decision_times = np.asarray(sa_settings["local_decision_time_ms"][:steps_executed], dtype=np.float64)
    valid_local_times = local_decision_times[np.isfinite(local_decision_times)]
    return {
        "decision_time_ms": local_decision_times.copy(),
        "decision_steps": np.flatnonzero(np.isfinite(local_decision_times)).astype(np.int64) + 1,
        "count": int(valid_local_times.size),
        "mean_ms": float(np.mean(valid_local_times)) if valid_local_times.size else np.nan,
        "max_ms": float(np.max(valid_local_times)) if valid_local_times.size else np.nan,
        "min_ms": float(np.min(valid_local_times)) if valid_local_times.size else np.nan,
    }


def _rotation_body_to_world(fish_state: np.ndarray) -> np.ndarray:
    """Build the body-to-world rotation matrix from the fish quaternion."""
    q0, q1, q2, q3 = fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
    return np.array(
        [
            [q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q0 * q1 + q3 * q2)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3],
        ],
        dtype=np.float64,
    )


def _append_controller_snapshot(sa_settings: dict, index: int, snapshot: dict | None) -> None:
    """Append one controller snapshot into the preallocated logging arrays."""
    fields = {
        "theta_ref": "theta_ref",
        "theta_act": "theta",
        "e_theta": "e_theta",
        "theta_int": "e_theta_int",
        "psi_ref": "psi_ref",
        "psi_act": "psi",
        "e_psi": "e_psi",
        "psi_int": "e_psi_int",
        "alpha5_ref": "alpha5_ref",
        "delta_ref": "delta_ref",
    }
    if snapshot is None:
        for key in fields:
            sa_settings[key][index] = np.nan
        return
    for key, source_key in fields.items():
        sa_settings[key][index] = float(snapshot.get(source_key, np.nan))


def _initial_log_arrays(cfg: HybridSimulationConfig) -> dict[str, np.ndarray | list]:
    """Allocate all simulation logging arrays in one place."""
    return {
        "cmd_vel_des": np.zeros((3, cfg.max_steps), dtype=np.float64),
        "vel_act": np.zeros((3, cfg.max_steps), dtype=np.float64),
        "theta_ref": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "theta_act": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "e_theta": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "theta_int": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "psi_ref": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "psi_act": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "e_psi": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "psi_int": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "alpha5_ref": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "delta_ref": np.full(cfg.max_steps, np.nan, dtype=np.float64),
        "mode": [],
        "local_planner_success": [],
        "local_decision_time_ms": np.full(cfg.max_steps, np.nan, dtype=np.float64),
    }


def _collect_visible_obstacles(curr_pos, robot_vel, all_true_obs, sensor_params):
    """Project sensor rays and recover the visible obstacle subset."""
    safe_zone = get_visible_obstacles(curr_pos, robot_vel, all_true_obs, sensor_params)
    end_pts = safe_zone["origin"].reshape(3, 1) + safe_zone["rays"] * safe_zone["dists"].reshape(1, -1)
    blocked_pts = end_pts[:, safe_zone["is_blocked"]].T.copy()
    visible_obs: list[dict] = []
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
    return blocked_pts, visible_obs, d_min


def _plot_result(result: dict, static_obs: list[dict], dyn_obs: list[dict], goal: np.ndarray, save_path: str | None, fps: int) -> None:
    """Render a single-planner result using the existing visualization module."""
    from simulation.main.visualization import plot_hybrid_result

    plot_hybrid_result(result, static_obs, dyn_obs, goal, save_path=save_path, fps=fps)


def _run_hybrid_simulation(
    planner_kind: str,
    local_planner,
    model_path=None,
    visualize: bool = False,
    animation_path: str | None = None,
    animation_fps: int = 20,
    controller_params_override: dict | None = None,
    sim_config: HybridSimulationConfig | None = None,
) -> dict:
    """Run the shared hybrid LOS simulation loop for either RL or MPC local planning."""
    del model_path
    cfg = HybridSimulationConfig() if sim_config is None else sim_config
    env_cfg = build_fish_env_config()
    scene = _build_hybrid_scene(cfg)
    traj_global = scene["traj_global"]
    fs = scene["fs"]
    fg = scene["fg"]
    scene_min = scene["scene_min"]
    scene_max = scene["scene_max"]
    los_params = scene["los_params"]
    static_obs = scene["static_obs"]
    dyn_obs = scene["dyn_obs"]
    sensor_params = scene["sensor_params"]
    sa_settings = _initial_log_arrays(cfg)

    s_max = traj_global["xyz"].shape[0]
    s_grid = np.arange(1, s_max + 1, dtype=np.float64)
    fx = lambda s: np.interp(s, s_grid, traj_global["xyz"][:, 0])
    fy = lambda s: np.interp(s, s_grid, traj_global["xyz"][:, 1])
    fz = lambda s: np.interp(s, s_grid, traj_global["xyz"][:, 2])

    fish_state = np.zeros(13, dtype=np.float64)
    fish_state[Q0 : Q3 + 1] = _quat_from_forward_vector(fs["v"])
    fish_state[[PX, PY, PZ]] = fs["p"]
    body_params = np.asarray(env_cfg["body_params"], dtype=np.float64).copy()
    fin_params = np.asarray(env_cfg["fin_params"], dtype=np.float64).copy()
    ctrl_state = _default_ctrl_state()
    hist = np.zeros(5, dtype=np.float64)
    c_a = float(env_cfg["c_A"])
    fin_f = float(env_cfg["fin_f"])
    fish_radius = float(env_cfg["fish_radius"])
    ref_min = np.asarray(env_cfg["ref_min"], dtype=np.float64)
    ref_max = np.asarray(env_cfg["ref_max"], dtype=np.float64)
    fish_params = {key: float(value) for key, value in env_cfg["controller_params"].items()}
    if controller_params_override:
        for key, value in controller_params_override.items():
            fish_params[key] = float(value)

    curr_pos = fs["p"].copy()
    robot_vel = fs["v"].copy()
    mode = "GLOBAL_TRACKING"
    current_path_idx = 1
    path_hist = [curr_pos.copy()]
    fish_state_hist = [fish_state.copy()]
    action_ref_hist = [np.zeros(5, dtype=np.float64)]
    fin_ref_step_hist: list[np.ndarray] = []
    fin_hist_in_step_hist: list[np.ndarray] = []
    step_time_hist: list[float] = []
    for obstacle in dyn_obs:
        _update_dynamic_obstacle(obstacle, 0.0)
    dyn_obs_hist = [np.asarray([obs["c"].copy() for obs in dyn_obs], dtype=np.float64)]
    blocked_pts_hist: list[np.ndarray] = [np.zeros((0, 3), dtype=np.float64)]
    los_target_hist = [np.full(3, np.nan, dtype=np.float64)]
    local_target_hist = [np.full(3, np.nan, dtype=np.float64)]
    title_hist = ["Starting Simulation with LOS Guidance..."]
    danger_dist_hist = [np.nan]
    reached_goal = False
    collided = False
    numerical_issue = False
    collision_info: dict | None = None

    for k in range(1, cfg.max_steps + 1):
        t_k = (k - 1) * cfg.dt
        for obstacle in dyn_obs:
            _update_dynamic_obstacle(obstacle, t_k + cfg.dt)
            obstacle["c"] = np.clip(obstacle["c"], scene_min + obstacle["r"], scene_max - obstacle["r"])

        all_true_obs = static_obs + dyn_obs
        blocked_pts, visible_obs, d_min = _collect_visible_obstacles(curr_pos, robot_vel, all_true_obs, sensor_params)
        enter_danger = np.isfinite(d_min) and d_min < sensor_params["danger_distance_enter"]
        exit_danger = np.isfinite(d_min) and d_min < sensor_params["danger_distance_exit"]
        if mode == "GLOBAL_TRACKING" and enter_danger:
            mode = "LOCAL_AVOIDANCE"
            local_planner.reset()
        elif mode == "LOCAL_AVOIDANCE" and not exit_danger:
            mode = "GLOBAL_TRACKING"

        sa_settings["mode"].append(mode)
        cmd_speed = _command_speed(cfg, t_k)
        los_target = np.full(3, np.nan, dtype=np.float64)
        local_target_log = np.full(3, np.nan, dtype=np.float64)
        controller_snapshot: dict | None = None
        local_success = True

        if mode == "LOCAL_AVOIDANCE":
            local_target = find_local_target(traj_global, curr_pos, cfg.local_lookahead)
            local_target_log = np.asarray(local_target, dtype=np.float64).copy()
            decision_tic = perf_counter()
            if planner_kind == "rl":
                action_ref, _, _, cmd_vel_global, ctrl_state = local_planner.plan(
                    fish_state,
                    hist,
                    local_target,
                    visible_obs,
                    ctrl_state,
                    cfg.dt,
                    fish_params,
                )
                controller_snapshot = ctrl_state
            else:
                attitude_ref, cmd_speed_ref, planner_info = local_planner.plan(
                    fish_state,
                    hist,
                    local_target,
                    visible_obs,
                    cmd_speed,
                )
                psi_ref = float(attitude_ref[0])
                theta_ref = float(attitude_ref[1])
                a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, ctrl_state = fin_controller(
                    psi_ref,
                    theta_ref,
                    cmd_speed_ref,
                    fish_state,
                    ctrl_state,
                    cfg.dt,
                    fish_params,
                )
                action_ref = np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64)
                controller_snapshot = ctrl_state
                cmd_vel_global = np.asarray(planner_info["cmd_vel_global"], dtype=np.float64)
                local_success = bool(planner_info.get("success", True))
            sa_settings["local_decision_time_ms"][k - 1] = (perf_counter() - decision_tic) * 1000.0
            sa_settings["cmd_vel_des"][:, k - 1] = cmd_vel_global
            title_hist.append(
                f"Step {k}: LOCAL AVOIDANCE ({planner_kind.upper()})"
                f"{'' if not np.isfinite(d_min) else f' - Visible Dist: {d_min:.2f}'}"
            )
        else:
            lookback = max(1, current_path_idx - cfg.lookback_margin)
            _, _, psi_ref, theta_ref = los_guidance_3d(
                curr_pos,
                (lookback, s_max),
                los_params["Delta"],
                los_params["Delta"],
                fx,
                fy,
                fz,
            )
            dists = np.sqrt(np.sum((traj_global["xyz"] - curr_pos.reshape(1, 3)) ** 2, axis=1))
            proj_idx = int(np.argmin(dists))
            current_path_idx = max(current_path_idx, proj_idx + 1)
            nearest_p = traj_global["xyz"][proj_idx]
            cmd_vel_global = np.array(
                [
                    cmd_speed * np.cos(theta_ref) * np.cos(psi_ref),
                    cmd_speed * np.cos(theta_ref) * np.sin(psi_ref),
                    -cmd_speed * np.sin(theta_ref),
                ],
                dtype=np.float64,
            )
            sa_settings["cmd_vel_des"][:, k - 1] = cmd_vel_global
            a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref, ctrl_state = fin_controller(
                psi_ref,
                theta_ref,
                cmd_speed,
                fish_state,
                ctrl_state,
                cfg.dt,
                fish_params,
            )
            action_ref = np.array([a1_ref, a2_ref, a3_ref, a4_ref, alpha5_ref], dtype=np.float64)
            controller_snapshot = ctrl_state
            los_target = np.array(
                [
                    nearest_p[0] + los_params["Delta"] * np.cos(theta_ref) * np.cos(psi_ref),
                    nearest_p[1] + los_params["Delta"] * np.cos(theta_ref) * np.sin(psi_ref),
                    nearest_p[2] - los_params["Delta"] * np.sin(theta_ref),
                ],
                dtype=np.float64,
            )
            title_hist.append(
                f"Step {k}: GLOBAL TRACKING (LOS) - "
                f"{'No obstacles in FOV' if not np.isfinite(d_min) else f'Visible Dist: {d_min:.2f}'}"
            )

        sa_settings["local_planner_success"].append(local_success)
        _append_controller_snapshot(sa_settings, k - 1, controller_snapshot)

        action_ref = np.clip(action_ref, ref_min, ref_max)
        fin_ref_step_hist.append(action_ref.copy())
        fin_hist_in_step_hist.append(hist.copy())
        step_time_hist.append(float(t_k))
        fish_state, hist = dynamics_step(fish_state, body_params, fin_params, cfg.dt, t_k, action_ref, hist, c_a, fin_f)
        if not np.isfinite(fish_state).all():
            numerical_issue = True
            blocked_pts_hist.append(blocked_pts)
            los_target_hist.append(los_target)
            local_target_hist.append(local_target_log)
            danger_dist_hist.append(d_min if np.isfinite(d_min) else np.nan)
            break

        curr_pos = fish_state[[PX, PY, PZ]].copy()
        rie = _rotation_body_to_world(fish_state)
        robot_vel = rie.T @ fish_state[[VX, VY, VZ]]
        sa_settings["vel_act"][:, k - 1] = robot_vel
        path_hist.append(curr_pos.copy())
        fish_state_hist.append(fish_state.copy())
        action_ref_hist.append(action_ref.copy())
        dyn_obs_hist.append(np.asarray([obs["c"].copy() for obs in dyn_obs], dtype=np.float64))
        blocked_pts_hist.append(blocked_pts)
        los_target_hist.append(los_target)
        local_target_hist.append(local_target_log)
        danger_dist_hist.append(d_min if np.isfinite(d_min) else np.nan)

        nearest_obstacle = nearest_obstacle_info(curr_pos, robot_vel, all_true_obs, fish_radius)
        collision_clearance = float(nearest_obstacle["clearance"])
        if collision_clearance <= 0.0:
            collided = True
            collision_info = {
                "step": k,
                "clearance": collision_clearance,
                "position": curr_pos.copy(),
                "obstacle_center": np.asarray(nearest_obstacle["c"], dtype=np.float64).copy(),
                "obstacle_radius": float(nearest_obstacle["r"]),
                "fish_radius": fish_radius,
                "distance": float(nearest_obstacle["distance"]),
            }
            title_hist[-1] = f"Step {k}: COLLISION - clearance {collision_clearance:.3f} m"
            break

        if np.linalg.norm(curr_pos - fg["p"]) < cfg.goal_threshold:
            reached_goal = True
            break

    steps_executed = len(path_hist) - 1
    result = {
        "path_hist": np.asarray(path_hist, dtype=np.float64),
        "fish_state_hist": np.asarray(fish_state_hist, dtype=np.float64),
        "action_ref_hist": np.asarray(action_ref_hist, dtype=np.float64),
        "fin_ref_step_hist": np.asarray(fin_ref_step_hist, dtype=np.float64),
        "fin_hist_in_step_hist": np.asarray(fin_hist_in_step_hist, dtype=np.float64),
        "step_time_hist": np.asarray(step_time_hist, dtype=np.float64),
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
        "sim_dt": float(cfg.dt),
        "c_a": c_a,
        "fin_f": fin_f,
        "reached_goal": reached_goal,
        "collided": collided,
        "numerical_issue": numerical_issue,
        "collision_info": collision_info,
        "goal": fg["p"].copy(),
        "final_state": fish_state,
        "final_hist": hist,
        "local_planner_kind": planner_kind,
    }
    result["local_planner_timing"] = _build_local_planner_timing(sa_settings, steps_executed)
    result["tracking_metrics"] = _compute_tracking_metrics(sa_settings, fish_params, steps_executed)
    if visualize:
        _plot_result(result, static_obs, dyn_obs, fg["p"], animation_path, animation_fps)
    return result


def run_hybrid_los_rl_simulation(
    model_path=None,
    visualize: bool = False,
    animation_path: str | None = None,
    animation_fps: int = 20,
    controller_params_override: dict | None = None,
    sim_config: HybridSimulationConfig | None = None,
) -> dict:
    """Run the hybrid simulation with the RL local planner."""
    planner = RLLocalPlanner(model_path=model_path, config=build_fish_env_config())
    return _run_hybrid_simulation(
        planner_kind="rl",
        local_planner=planner,
        model_path=model_path,
        visualize=visualize,
        animation_path=animation_path,
        animation_fps=animation_fps,
        controller_params_override=controller_params_override,
        sim_config=sim_config,
    )


def run_hybrid_los_mpc_simulation(
    visualize: bool = False,
    animation_path: str | None = None,
    animation_fps: int = 20,
    controller_params_override: dict | None = None,
    sim_config: HybridSimulationConfig | None = None,
) -> dict:
    """Run the hybrid simulation with the MPC local planner."""
    planner = MPCLocalPlanner(config=build_fish_env_config())
    return _run_hybrid_simulation(
        planner_kind="mpc",
        local_planner=planner,
        visualize=visualize,
        animation_path=animation_path,
        animation_fps=animation_fps,
        controller_params_override=controller_params_override,
        sim_config=sim_config,
    )


def run_hybrid_los_comparison_simulation(
    model_path=None,
    visualize: bool = False,
    animation_path: str | None = None,
    animation_fps: int = 20,
    controller_params_override: dict | None = None,
    sim_config: HybridSimulationConfig | None = None,
) -> dict:
    """Run RL and MPC under the same scene and return both results."""
    cfg = HybridSimulationConfig() if sim_config is None else sim_config
    rl_result = run_hybrid_los_rl_simulation(
        model_path=model_path,
        visualize=False,
        controller_params_override=controller_params_override,
        sim_config=cfg,
    )
    mpc_result = run_hybrid_los_mpc_simulation(
        visualize=False,
        controller_params_override=controller_params_override,
        sim_config=cfg,
    )
    comparison = {"rl": rl_result, "mpc": mpc_result}
    if visualize:
        from simulation.main.visualization import plot_hybrid_comparison_result

        static_obs, _ = _build_default_obstacles()
        plot_hybrid_comparison_result(
            rl_result,
            mpc_result,
            static_obs,
            rl_result["goal"],
            save_path=animation_path,
            fps=animation_fps,
        )
    return comparison
