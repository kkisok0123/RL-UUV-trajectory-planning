from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

# Allow `python simulation/main.py` from the repo root or an IDE run config.
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dynamics_wrapper import probe_fin_wrenches as dynamics_probe_fin_wrenches
from dynamics_wrapper import step as dynamics_step
from dynamics_wrapper import step_with_body_wrench as dynamics_step_with_body_wrench
from dynamics_wrapper.params import load_body_params, load_fin_params
from dynamics_wrapper.indices import PX, PY, PZ, Q0, Q1, Q2, Q3
from rl.local_planner import RLLocalPlanner
from rl.configs.fish_env import build_fish_env_config
from simulation.global_planning.los import los_guidance_3d
from simulation.global_planning.plan_global_bezier_pso import planGlobalBezierPSO
from simulation.local_planning.fin_controller import (
    run_layer1_controller,
    run_layer2_controller,
)
from simulation.local_planning.path_utils import find_local_target
from simulation.local_planning.sensor import get_visible_obstacles


@dataclass
class HybridSimulationConfig:
    dt: float = 0.25
    max_steps: int = 10000
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


def run_hybrid_los_rl_simulation(
    model_path=None,
    visualize: bool = False,
    control_mode: str = "layer2",
) -> dict:
    mode_aliases = {
        "full": "layer2",
        "layer1_direct_wrench": "layer1",
    }
    control_mode = mode_aliases.get(control_mode, control_mode)
    valid_modes = {"layer1", "layer2"}
    if control_mode not in valid_modes:
        raise ValueError(f"control_mode must be one of {sorted(valid_modes)}")

    cfg = HybridSimulationConfig()
    env_cfg = build_fish_env_config()
    sa_settings: dict[str, np.ndarray] = {}

    fs = {"p": np.array([0.0, 0.0, 0.0]), "v": np.array([1.0, 3.0, 2.0]), "a": np.zeros(3)}
    fg = {"p": np.array([15.0, 15.0, 2.0]), "v": np.zeros(3), "a": np.zeros(3)}

    opts = {
        "t0": 0.0,
        "t3": 15.0,
        "vmax": 0.6,
        "amax": 1.5,
        "N": 300,
        "psoM": 200,
        "psoT": 400,
        "w": 0.7,
        "c1": 1.6,
        "c2": 1.6,
        "clearance": 0.2,
        "Mk_obs": 1e3,
        "Mk_speed": 1e2,
        "Mk_acc": 1e2,
        "Mk_kappa": 1e2,
        "dt": cfg.dt,
    }

    los_params = {"Delta": 2.5, "k_p": 0.5}
    static_obs = [
        _make_obstacle([5.0, 5.0, 0.0], 1.6),
        _make_obstacle([12.0, 13.0, 1.0], 1.5),
    ]
    dyn_obs = [_make_obstacle([12.0, 2.0, 0.5], 1.5, [-0.5, 0.6, 0.3])]

    traj_global = planGlobalBezierPSO(fs, fg, opts, static_obs + dyn_obs)
    s_max = traj_global["xyz"].shape[0]
    fx = lambda s: np.interp(s, np.arange(1, s_max + 1), traj_global["xyz"][:, 0])
    fy = lambda s: np.interp(s, np.arange(1, s_max + 1), traj_global["xyz"][:, 1])
    fz = lambda s: np.interp(s, np.arange(1, s_max + 1), traj_global["xyz"][:, 2])

    sensor_params = {
        "range": 5.0,
        "fov_angle": 45.0,
        "danger_distance": 3.0,
        "dt": cfg.dt,
        "num_rays": 100,
    }

    fish_state = np.zeros(13, dtype=np.float64)
    fish_state[:3] = [0,0,0]
    fish_state[Q0] = 1.0
    fish_state[[PX, PY, PZ]] = fs["p"]
    body_params = load_body_params()
    fin_params = load_fin_params()

    ctrl_state: dict = {}
    hist = np.asarray(env_cfg["trim_refs"], dtype=np.float64).copy()
    c_A = 5.0
    fin_f = 4.0
    ref_min = np.asarray(env_cfg["ref_min"], dtype=np.float64)
    ref_max = np.asarray(env_cfg["ref_max"], dtype=np.float64)
    fish_controller_params = deepcopy(env_cfg["controller_params"])

    local_planner = RLLocalPlanner(model_path=model_path)
    curr_pos = fs["p"].copy()
    robot_vel = [0,0,0]
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
    sa_settings["wrench_cmd_des"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["actuator_cmd_des"] = np.zeros((5, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_des_total"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_des_right"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_des_left"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_des_tail"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_real_total"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_real_right"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_real_left"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["fin_wrench_real_tail"] = np.zeros((6, cfg.max_steps), dtype=np.float64)
    sa_settings["vel_act"] = np.zeros((3, cfg.max_steps), dtype=np.float64)
    sa_settings["mode"] = []
    sa_settings["control_mode"] = control_mode

    for k in range(1, cfg.max_steps + 1):
        for obstacle in dyn_obs:
            obstacle["c"] = obstacle["c"] + obstacle["v"] * cfg.dt
            bounds = (0.0, 15.0)
            if obstacle["c"][0] < bounds[0] or obstacle["c"][0] > bounds[1]:
                obstacle["v"][0] = -obstacle["v"][0]
            if obstacle["c"][1] < bounds[0] or obstacle["c"][1] > bounds[1]:
                obstacle["v"][1] = -obstacle["v"][1]

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
        body_wrench_cmd = np.zeros(6, dtype=np.float64)
        action_ref = np.zeros(5, dtype=np.float64)

        if mode == "LOCAL_AVOIDANCE":
            local_target = find_local_target(traj_global, curr_pos, cfg.local_lookahead)
            local_target_log = np.asarray(local_target, dtype=np.float64).copy()
            if control_mode == "layer1":
                body_wrench_cmd, _, _, _, cmd_vel_global, ctrl_state = local_planner.plan_layer1_wrench(
                    fish_state,
                    hist,
                    local_target,
                    visible_obs,
                    ctrl_state,
                    cfg.dt,
                    fish_controller_params,
                )
            else:
                action_ref, _, _, _, cmd_vel_global, ctrl_state = local_planner.plan_layer2_actuator(
                    fish_state,
                    hist,
                    local_target,
                    visible_obs,
                    ctrl_state,
                    cfg.dt,
                    fish_controller_params,
                )
            sa_settings["cmd_vel_des"][:, k - 1] = cmd_vel_global
            if np.isfinite(d_min):
                title_hist.append(f"Step {k}: LOCAL AVOIDANCE ({control_mode}) - Visible Dist: {d_min:.2f}")
            else:
                title_hist.append(f"Step {k}: LOCAL AVOIDANCE ({control_mode})")
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
                    opts["vmax"] * np.cos(theta_ref) * np.cos(psi_ref),
                    opts["vmax"] * np.cos(theta_ref) * np.sin(psi_ref),
                    -opts["vmax"] * np.sin(theta_ref),
                ],
                dtype=np.float64,
            )
            ref_cmd = np.array([opts["vmax"], psi_ref, theta_ref], dtype=np.float64)
            sa_settings["cmd_vel_des"][:, k - 1] = cmd_vel_global
            if control_mode == "layer1":
                body_wrench_cmd, ctrl_state = run_layer1_controller(
                    ref_cmd,
                    fish_state,
                    ctrl_state,
                    cfg.dt,
                    fish_controller_params,
                )
            else:
                action_ref, ctrl_state = run_layer2_controller(
                    ref_cmd,
                    fish_state,
                    hist,
                    ctrl_state,
                    cfg.dt,
                    fish_controller_params,
                )
            los_target = np.array(
                [
                    nearest_p[0] + los_params["Delta"] * np.cos(theta_ref) * np.cos(psi_ref),
                    nearest_p[1] + los_params["Delta"] * np.cos(theta_ref) * np.sin(psi_ref),
                    nearest_p[2] - los_params["Delta"] * np.sin(theta_ref),
                ],
                dtype=np.float64,
            )
            title_hist.append(
                f"Step {k}: GLOBAL TRACKING ({control_mode}) - {'No obstacles in FOV' if not np.isfinite(d_min) else f'Visible Dist: {d_min:.2f}'}"
            )

        if control_mode == "layer1":
            sa_settings["wrench_cmd_des"][:, k - 1] = body_wrench_cmd
            fish_state, hist = dynamics_step_with_body_wrench(
                fish_state,
                body_params,
                cfg.dt,
                body_wrench_cmd,
                hist,
            )
        else:
            action_ref = np.clip(np.asarray(action_ref, dtype=np.float64).reshape(5), ref_min, ref_max)
            sa_settings["actuator_cmd_des"][:, k - 1] = action_ref
            if "tau_target" in ctrl_state:
                sa_settings["wrench_cmd_des"][:, k - 1] = np.asarray(ctrl_state["tau_target"], dtype=np.float64).reshape(6)
                sa_settings["fin_wrench_des_total"][:, k - 1] = np.asarray(ctrl_state["tau_target"], dtype=np.float64).reshape(6)
            if all(key in ctrl_state for key in ("w_r_des", "w_l_des", "w_t_des")):
                sa_settings["fin_wrench_des_right"][:, k - 1] = np.asarray(ctrl_state["w_r_des"], dtype=np.float64).reshape(6)
                sa_settings["fin_wrench_des_left"][:, k - 1] = np.asarray(ctrl_state["w_l_des"], dtype=np.float64).reshape(6)
                sa_settings["fin_wrench_des_tail"][:, k - 1] = np.asarray(ctrl_state["w_t_des"], dtype=np.float64).reshape(6)
            probed_fin_wrenches = dynamics_probe_fin_wrenches(
                fish_state,
                fin_params,
                t_k,
                action_ref,
                hist,
                c_A,
                fin_f,
            )
            sa_settings["fin_wrench_real_total"][:, k - 1] = probed_fin_wrenches["total"]
            sa_settings["fin_wrench_real_right"][:, k - 1] = probed_fin_wrenches["right"]
            sa_settings["fin_wrench_real_left"][:, k - 1] = probed_fin_wrenches["left"]
            sa_settings["fin_wrench_real_tail"][:, k - 1] = probed_fin_wrenches["tail"]
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
        "dt": cfg.dt,
        "sensor_params": sensor_params,
        "initial_robot_vel": fs["v"].copy(),
        "reached_goal": reached_goal,
        "numerical_issue": numerical_issue,
        "goal": fg["p"].copy(),
        "final_state": fish_state,
        "final_hist": hist,
    }
    if visualize:
        from simulation.visualization import plot_hybrid_result

        plot_hybrid_result(result, static_obs, dyn_obs, fg["p"])
    return result


def main() -> None:
    result = run_hybrid_los_rl_simulation(visualize=True, control_mode="layer2")
    print("reached_goal:", result["reached_goal"])
    print("path_len:", len(result["path_hist"]))


if __name__ == "__main__":
    main()
