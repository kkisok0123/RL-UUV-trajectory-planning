from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dynamics.indices import A1, A2, A3, A4, A5, PX, PY, PZ, Q0, Q1, Q2, Q3
from rl_training.envs.geometry import quaternion_to_rotation_matrix

_LIVE_ANIMATIONS: list[FuncAnimation] = []


def _save_animation(anim: FuncAnimation, save_path: str, fps: int = 20) -> None:
    suffix = str(save_path).lower()
    if suffix.endswith(".gif"):
        writer = PillowWriter(fps=int(fps))
    elif suffix.endswith(".mp4"):
        writer = FFMpegWriter(fps=int(fps))
    else:
        raise ValueError("Animation save path must end with .gif or .mp4")
    anim.save(save_path, writer=writer)


def _sphere_mesh(center: np.ndarray, radius: float, u: np.ndarray, v: np.ndarray):
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return x, y, z


def _rotation_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return points @ rotation.T + translation


def _body_face_vertices(vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64)
    return [vertices[face] for face in faces]


def _build_fish_geometry() -> dict:
    body_length1 = 1.5
    body_width1 = 1.0
    body_height1 = 0.3
    body_length2 = 0.3
    body_width2 = 0.6
    body_height2 = 0.1

    x_point = np.array(
        [
            body_length1 / 2 + body_length2,
            body_length1 / 2 + body_length2,
            body_length1 / 2 + body_length2,
            body_length1 / 2 + body_length2,
            body_length1 / 2,
            body_length1 / 2,
            body_length1 / 2,
            body_length1 / 2,
            -body_length1 / 2,
            -body_length1 / 2,
            -body_length1 / 2,
            -body_length1 / 2,
            -body_length1 / 2 - body_length2,
            -body_length1 / 2 - body_length2,
            -body_length1 / 2 - body_length2,
            -body_length1 / 2 - body_length2,
        ],
        dtype=np.float64,
    )
    y_point = np.array(
        [
            body_width2 / 2,
            -body_width2 / 2,
            -body_width2 / 2,
            body_width2 / 2,
            body_width1 / 2,
            -body_width1 / 2,
            -body_width1 / 2,
            body_width1 / 2,
            body_width1 / 2,
            -body_width1 / 2,
            -body_width1 / 2,
            body_width1 / 2,
            body_width2 / 2,
            -body_width2 / 2,
            -body_width2 / 2,
            body_width2 / 2,
        ],
        dtype=np.float64,
    )
    z_point = np.array(
        [
            -body_height2 / 2,
            -body_height2 / 2,
            body_height2 / 2,
            body_height2 / 2,
            -body_height1 / 2,
            -body_height1 / 2,
            body_height1 / 2,
            body_height1 / 2,
            -body_height1 / 2,
            -body_height1 / 2,
            body_height1 / 2,
            body_height1 / 2,
            -body_height2 / 2,
            -body_height2 / 2,
            body_height2 / 2,
            body_height2 / 2,
        ],
        dtype=np.float64,
    )
    body_vertices = np.column_stack([x_point, y_point, z_point])
    body_faces = np.array(
        [
            [1, 2, 3, 4],
            [3, 4, 8, 7],
            [8, 7, 11, 12],
            [11, 12, 16, 15],
            [13, 14, 15, 16],
            [1, 2, 6, 5],
            [6, 5, 9, 10],
            [9, 10, 14, 13],
            [1, 4, 8, 5],
            [8, 5, 9, 12],
            [9, 12, 16, 13],
            [2, 3, 7, 6],
            [7, 6, 10, 11],
            [10, 11, 15, 14],
        ],
        dtype=np.int64,
    ) - 1

    fin_offset_y = 0.1
    tail_offset_x = 0.1

    def c_right(y: np.ndarray) -> np.ndarray:
        return (
            0.00147
            - 1.03112 * y
            + 4.83404 * y**2
            - 39.9661 * y**3
            + 177.53522 * y**4
            - 455.81824 * y**5
            + 698.03875 * y**6
            - 628.68641 * y**7
            + 306.97937 * y**8
            - 62.67097 * y**9
            + 0.4
        )

    def d_right(y: np.ndarray) -> np.ndarray:
        return (
            -0.97549
            + 0.45979 * y
            - 1.99058 * y**2
            + 10.83717 * y**3
            - 47.03231 * y**4
            + 122.6193 * y**5
            - 191.79506 * y**6
            + 176.78374 * y**7
            - 88.49801 * y**8
            + 18.55854 * y**9
            + 0.4
        )

    def c_left(y: np.ndarray) -> np.ndarray:
        return c_right(-y)

    def d_left(y: np.ndarray) -> np.ndarray:
        return d_right(-y)

    def e_tail(y: np.ndarray) -> np.ndarray:
        return (
            -0.43996
            + 0.00142 * y
            + 3.3867 * y**2
            + 0.05211 * y**3
            + 7.4239 * y**4
            - 0.32164 * y**5
            - 116.92926 * y**6
            - 5.64292 * y**7
            + 313.05641 * y**8
            + 39.1549 * y**9
        )

    yy_right = np.arange(0.0, 1.15 + 1e-9, 0.01, dtype=np.float64)
    xx_1_right = c_right(yy_right)
    xx_2_right = d_right(yy_right)
    right_fin = np.column_stack(
        [
            np.concatenate([xx_1_right, np.flip(xx_2_right)]),
            np.concatenate([yy_right + fin_offset_y, np.flip(yy_right) + fin_offset_y]),
            np.zeros(2 * yy_right.size, dtype=np.float64),
        ]
    )

    yy_left = np.arange(-1.15, 0.0 + 1e-9, 0.01, dtype=np.float64)
    xx_1_left = c_left(yy_left)
    xx_2_left = d_left(yy_left)
    left_fin = np.column_stack(
        [
            np.concatenate([xx_1_left, np.flip(xx_2_left)]),
            np.concatenate([yy_left - fin_offset_y, np.flip(yy_left) - fin_offset_y]),
            np.zeros(2 * yy_left.size, dtype=np.float64),
        ]
    )

    yy_tail = np.arange(-0.3875, 0.3875 + 1e-9, 0.01, dtype=np.float64)
    tail_fin = np.column_stack(
        [
            e_tail(yy_tail) - tail_offset_x,
            yy_tail,
            np.zeros(yy_tail.size, dtype=np.float64),
        ]
    )

    return {
        "body_vertices": body_vertices,
        "body_faces": body_faces,
        "right_fin_vertices": right_fin,
        "left_fin_vertices": left_fin,
        "tail_fin_vertices": tail_fin,
        "right_attach": np.array([0.0, body_width1 / 2, 0.0], dtype=np.float64),
        "left_attach": np.array([0.0, -body_width1 / 2, 0.0], dtype=np.float64),
        "tail_attach": np.array([-body_length1 / 2 - body_length2, 0.0, 0.0], dtype=np.float64),
        "fin_offset_y": float(fin_offset_y),
        "tail_offset_x": float(tail_offset_x),
    }


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _interp_fish_state(fish_state_hist: np.ndarray, seg_idx: int, tau: float) -> np.ndarray:
    fish_state_hist = np.asarray(fish_state_hist, dtype=np.float64)
    if fish_state_hist.ndim != 2 or fish_state_hist.shape[0] == 0:
        fallback = np.zeros(PZ + 1, dtype=np.float64)
        fallback[Q0] = 1.0
        return fallback
    if seg_idx >= fish_state_hist.shape[0] - 1:
        return fish_state_hist[-1].copy()
    state0 = fish_state_hist[seg_idx]
    state1 = fish_state_hist[seg_idx + 1]
    interp = (1.0 - tau) * state0 + tau * state1
    interp[Q0 : Q3 + 1] = _normalize_quaternion(interp[Q0 : Q3 + 1])
    return interp


def _cpg_amplitude(ref: np.ndarray, hist_in: np.ndarray, c_a: float, t_local: float) -> np.ndarray:
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    hist_in = np.asarray(hist_in, dtype=np.float64).reshape(-1)
    half_c = 0.5 * float(c_a)
    return ref + (hist_in - ref) * (1.0 + half_c * t_local) * np.exp(-half_c * t_local)


def _compute_fin_angles(ref: np.ndarray, hist_in: np.ndarray, c_a: float, fin_f: float, t_k: float, t_local: float) -> np.ndarray:
    amps = _cpg_amplitude(ref, hist_in, c_a, t_local)
    phase = 2.0 * np.pi * float(fin_f) * (float(t_k) + float(t_local))
    sin_phase = float(np.sin(phase))
    cos_phase = float(np.cos(phase))
    return np.array(
        [
            amps[A1] * sin_phase,
            -amps[A2] * cos_phase,
            -amps[A3] * sin_phase,
            -amps[A4] * cos_phase,
            amps[A5],
        ],
        dtype=np.float64,
    )


def _right_fin_rotation(alpha1: float, alpha2: float) -> np.ndarray:
    s1, c1 = float(np.sin(alpha1)), float(np.cos(alpha1))
    s2, c2 = float(np.sin(alpha2)), float(np.cos(alpha2))
    return np.array(
        [[c2, s1 * s2, -c1 * s2], [0.0, c1, s1], [s2, -s1 * c2, c1 * c2]],
        dtype=np.float64,
    )


def _left_fin_rotation(alpha3: float, alpha4: float) -> np.ndarray:
    s3, c3 = float(np.sin(alpha3)), float(np.cos(alpha3))
    s4, c4 = float(np.sin(alpha4)), float(np.cos(alpha4))
    return np.array(
        [[c4, s3 * s4, -c3 * s4], [0.0, c3, s3], [s4, -s3 * c4, c3 * c4]],
        dtype=np.float64,
    )


def _tail_fin_rotation(alpha5: float) -> np.ndarray:
    s5, c5 = float(np.sin(alpha5)), float(np.cos(alpha5))
    return np.array([[c5, 0.0, -s5], [0.0, 1.0, 0.0], [s5, 0.0, c5]], dtype=np.float64)


def _build_cone_mesh(range_max: float, fov_deg: float, n_pts: int = 20):
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts + 1)
    x = np.array([0.0, range_max], dtype=np.float64)
    radius = np.array([0.0, range_max * np.tan(np.deg2rad(fov_deg))], dtype=np.float64)
    theta_grid, x_grid = np.meshgrid(theta, x, indexing="xy")
    radius_grid = np.broadcast_to(radius[:, None], x_grid.shape)
    y_grid = radius_grid * np.cos(theta_grid)
    z_grid = radius_grid * np.sin(theta_grid)
    return x_grid, y_grid, z_grid


def _cone_rotation(velocity: np.ndarray) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(velocity))
    if speed > 1e-2:
        v_dir = velocity / speed
    else:
        v_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    base_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rot_axis = np.cross(base_axis, v_dir)
    sin_ang = float(np.linalg.norm(rot_axis))
    cos_ang = float(np.dot(base_axis, v_dir))

    if sin_ang < 1e-6:
        if cos_ang > 0.0:
            return np.eye(3, dtype=np.float64)
        rot = -np.eye(3, dtype=np.float64)
        rot[1, 1] = 1.0
        rot[2, 2] = -1.0
        return rot

    u = rot_axis / sin_ang
    k_mat = np.array(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + sin_ang * k_mat + (1.0 - cos_ang) * (k_mat @ k_mat)


def _transform_cone(
    cone_x_base: np.ndarray,
    cone_y_base: np.ndarray,
    cone_z_base: np.ndarray,
    center: np.ndarray,
    velocity: np.ndarray,
):
    pts_base = np.vstack([cone_x_base.ravel(), cone_y_base.ravel(), cone_z_base.ravel()])
    pts_rot = _cone_rotation(velocity) @ pts_base
    x = pts_rot[0].reshape(cone_x_base.shape) + center[0]
    y = pts_rot[1].reshape(cone_y_base.shape) + center[1]
    z = pts_rot[2].reshape(cone_z_base.shape) + center[2]
    return x, y, z


def _axis_limits_from_points(
    points: list[np.ndarray],
    min_span: float = 6.0,
    pad_ratio: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    valid_points = []
    for pts in points:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.size == 0:
            continue
        pts = pts.reshape(-1, 3)
        mask = np.isfinite(pts).all(axis=1)
        if np.any(mask):
            valid_points.append(pts[mask])

    if not valid_points:
        mins = np.array([-3.0, -3.0, -3.0], dtype=np.float64)
        maxs = np.array([3.0, 3.0, 3.0], dtype=np.float64)
        return mins, maxs

    all_points = np.vstack(valid_points)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    spans = np.maximum(maxs - mins, min_span)
    pads = np.maximum(spans * pad_ratio, 0.5)
    centers = 0.5 * (mins + maxs)
    half_spans = 0.5 * spans + pads
    return centers - half_spans, centers + half_spans


def _set_3d_axis_scale(
    ax,
    axis_mins: np.ndarray,
    axis_maxs: np.ndarray,
) -> None:
    axis_mins = np.asarray(axis_mins, dtype=np.float64).reshape(3)
    axis_maxs = np.asarray(axis_maxs, dtype=np.float64).reshape(3)
    axis_spans = np.maximum(axis_maxs - axis_mins, 1.0)
    ax.set_xlim(float(axis_mins[0]), float(axis_maxs[0]))
    ax.set_ylim(float(axis_mins[1]), float(axis_maxs[1]))
    ax.set_zlim(float(axis_mins[2]), float(axis_maxs[2]))
    try:
        ax.set_box_aspect(tuple(float(span) for span in axis_spans))
    except AttributeError:
        pass


def _unwrap_angle_series(angle_series: np.ndarray) -> np.ndarray:
    angle_series = np.asarray(angle_series, dtype=np.float64).copy()
    if angle_series.ndim != 1 or angle_series.size == 0:
        return angle_series

    finite_mask = np.isfinite(angle_series)
    if not np.any(finite_mask):
        return angle_series

    finite_indices = np.flatnonzero(finite_mask)
    split_points = np.where(np.diff(finite_indices) > 1)[0] + 1
    for run_indices in np.split(finite_indices, split_points):
        angle_series[run_indices] = np.unwrap(angle_series[run_indices])
    return angle_series


def plot_hybrid_result(
    result: dict,
    static_obs: list[dict],
    dyn_obs: list[dict],
    goal: np.ndarray,
    save_path: str | None = None,
    fps: int = 20,
) -> None:
    del dyn_obs

    traj_global = np.asarray(result["traj_global"]["xyz"], dtype=np.float64)
    path_hist = np.asarray(result["path_hist"], dtype=np.float64)
    valid_mask = np.isfinite(path_hist).all(axis=1)
    valid_path = path_hist[valid_mask]
    if valid_path.size == 0:
        valid_path = np.zeros((1, 3), dtype=np.float64)

    steps_executed = int(result.get("steps_executed", max(len(valid_path) - 1, 0)))
    dyn_obs_hist = np.asarray(result.get("dyn_obs_hist", []), dtype=np.float64)
    dyn_obs_radii = np.asarray(result.get("dyn_obs_radii", []), dtype=np.float64)
    blocked_pts_hist = result.get("blocked_pts_hist", [])
    los_target_hist = np.asarray(result.get("los_target_hist", []), dtype=np.float64)
    local_target_hist = np.asarray(result.get("local_target_hist", []), dtype=np.float64)
    title_hist = list(result.get("title_hist", []))
    danger_dist_hist = np.asarray(result.get("danger_dist_hist", []), dtype=np.float64)
    sensor_params = result.get("sensor_params", {"range": 5.0, "fov_angle": 45.0})
    settings = result.get("settings", {})
    mode_hist = list(settings.get("mode", []))[:steps_executed]
    vel_act = np.asarray(settings.get("vel_act", np.zeros((3, steps_executed))), dtype=np.float64)
    cmd_vel_des = np.asarray(
        settings.get("cmd_vel_des", np.zeros((3, steps_executed))), dtype=np.float64
    )
    theta_ref = np.asarray(settings.get("theta_ref", np.full(steps_executed, np.nan)), dtype=np.float64)
    theta_act = np.asarray(settings.get("theta_act", np.full(steps_executed, np.nan)), dtype=np.float64)
    e_theta = np.asarray(settings.get("e_theta", np.full(steps_executed, np.nan)), dtype=np.float64)
    psi_ref = np.asarray(settings.get("psi_ref", np.full(steps_executed, np.nan)), dtype=np.float64)
    psi_act = np.asarray(settings.get("psi_act", np.full(steps_executed, np.nan)), dtype=np.float64)
    e_psi = np.asarray(settings.get("e_psi", np.full(steps_executed, np.nan)), dtype=np.float64)
    alpha5_ref = np.asarray(settings.get("alpha5_ref", np.full(steps_executed, np.nan)), dtype=np.float64)
    delta_ref = np.asarray(settings.get("delta_ref", np.full(steps_executed, np.nan)), dtype=np.float64)
    fish_state_hist = np.asarray(result.get("fish_state_hist", []), dtype=np.float64)
    fin_ref_step_hist = np.asarray(result.get("fin_ref_step_hist", []), dtype=np.float64)
    fin_hist_in_step_hist = np.asarray(result.get("fin_hist_in_step_hist", []), dtype=np.float64)
    step_time_hist = np.asarray(result.get("step_time_hist", []), dtype=np.float64)
    initial_robot_vel = np.asarray(
        result.get("initial_robot_vel", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        dtype=np.float64,
    )
    fish_geom = _build_fish_geometry()
    sim_dt = float(result.get("sim_dt", sensor_params.get("dt", 0.2)))
    fin_f = float(result.get("fin_f", 1.0))
    c_a = float(result.get("c_a", 0.0))

    if dyn_obs_hist.ndim == 3 and dyn_obs_hist.shape[0] > len(valid_path):
        dyn_obs_hist = dyn_obs_hist[: len(valid_path)]
    if dyn_obs_hist.ndim != 3:
        dyn_obs_hist = np.zeros((len(valid_path), 0, 3), dtype=np.float64)
    if dyn_obs_radii.ndim != 1:
        dyn_obs_radii = np.zeros((0,), dtype=np.float64)

    bounds_points: list[np.ndarray] = [
        traj_global,
        valid_path,
        np.asarray(goal, dtype=np.float64).reshape(1, 3),
    ]
    if static_obs:
        static_bounds = []
        for obstacle in static_obs:
            center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
            radius = float(obstacle["r"])
            static_bounds.extend([center - radius, center + radius])
        bounds_points.append(np.asarray(static_bounds, dtype=np.float64))
    if dyn_obs_hist.size > 0 and dyn_obs_hist.shape[1] > 0:
        for obs_idx in range(dyn_obs_hist.shape[1]):
            radius = float(dyn_obs_radii[obs_idx]) if obs_idx < dyn_obs_radii.size else 0.0
            bounds_points.append(dyn_obs_hist[:, obs_idx, :] - radius)
            bounds_points.append(dyn_obs_hist[:, obs_idx, :] + radius)
    axis_mins, axis_maxs = _axis_limits_from_points(bounds_points, min_span=2.0, pad_ratio=0.08)
    fig = plt.figure(
        figsize=(12, 10),
        facecolor="white",
    )
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Hybrid Trajectory Planning with LOS Guidance")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    _set_3d_axis_scale(ax, axis_mins, axis_maxs)
    ax.set_autoscale_on(False)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.view_init(elev=30.0, azim=-37.5)
    ax.grid(True)

    ax.plot(
        traj_global[:, 0],
        traj_global[:, 1],
        traj_global[:, 2],
        "k--",
        linewidth=1.5,
        label="Global Bezier Path",
    )

    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 12j]
    for obstacle in static_obs:
        x, y, z = _sphere_mesh(np.asarray(obstacle["c"], dtype=np.float64), float(obstacle["r"]), u, v)
        ax.plot_surface(
            x,
            y,
            z,
            color=(0.8, 0.2, 0.2),
            alpha=0.6,
            linewidth=0.0,
            edgecolor="none",
        )

    dyn_surfaces = []
    dyn_trails = []
    if dyn_obs_hist.shape[1] > 0:
        for obs_idx in range(dyn_obs_hist.shape[1]):
            x, y, z = _sphere_mesh(dyn_obs_hist[0, obs_idx], float(dyn_obs_radii[obs_idx]), u, v)
            dyn_surfaces.append(
                ax.plot_surface(
                    x,
                    y,
                    z,
                    color=(0.2, 0.2, 0.9),
                    alpha=0.6,
                    linewidth=0.0,
                    edgecolor="none",
                )
            )
            trail, = ax.plot(
                dyn_obs_hist[:1, obs_idx, 0],
                dyn_obs_hist[:1, obs_idx, 1],
                dyn_obs_hist[:1, obs_idx, 2],
                color=(0.2, 0.2, 0.9),
                linewidth=1.2,
                alpha=0.8,
            )
            dyn_trails.append(trail)

    goal_marker = ax.scatter(
        [goal[0]],
        [goal[1]],
        [goal[2]],
        c="y",
        s=180,
        marker="p",
        edgecolors="k",
        label="Goal",
    )

    render_substeps = 1
    # Exporting every fin-motion substep makes GIF generation dominate runtime.
    # if steps_executed > 0 and sim_dt > 1e-9 and fin_f > 1e-9:
    if save_path is None and steps_executed > 0 and sim_dt > 1e-9 and fin_f > 1e-9:
        render_substeps = max(1, int(np.ceil(8.0 * sim_dt * fin_f)))
    render_frames = max(1, steps_executed * render_substeps + 1)

    def _render_segment(render_idx: int) -> tuple[int, float]:
        if steps_executed <= 0:
            return 0, 0.0
        if render_idx >= render_frames - 1:
            return steps_executed, 0.0
        seg_idx = min(render_idx // render_substeps, steps_executed - 1)
        tau = (render_idx % render_substeps) / render_substeps
        return seg_idx, tau

    def _interp_series(series: np.ndarray, seg_idx: int, tau: float) -> np.ndarray:
        series = np.asarray(series, dtype=np.float64)
        if series.ndim != 2 or series.shape[0] == 0:
            return np.full(3, np.nan, dtype=np.float64)
        if seg_idx >= series.shape[0] - 1:
            return series[min(seg_idx, series.shape[0] - 1)].copy()
        return (1.0 - tau) * series[seg_idx] + tau * series[seg_idx + 1]

    def _get_render_velocity(seg_idx: int, tau: float) -> np.ndarray:
        if vel_act.ndim != 2 or vel_act.shape[1] == 0:
            return initial_robot_vel
        idx0 = max(0, min(seg_idx - 1, vel_act.shape[1] - 1))
        idx1 = max(0, min(seg_idx, vel_act.shape[1] - 1))
        if seg_idx <= 0:
            v0 = initial_robot_vel
        else:
            v0 = vel_act[:, idx0]
        v1 = vel_act[:, idx1]
        return (1.0 - tau) * v0 + tau * v1

    def _get_fin_angles(seg_idx: int, tau: float) -> np.ndarray:
        if seg_idx >= steps_executed or fin_ref_step_hist.ndim != 2 or fin_hist_in_step_hist.ndim != 2:
            return np.zeros(A5 + 1, dtype=np.float64)
        refs = fin_ref_step_hist[min(seg_idx, fin_ref_step_hist.shape[0] - 1)]
        hist_in = fin_hist_in_step_hist[min(seg_idx, fin_hist_in_step_hist.shape[0] - 1)]
        t_k = float(step_time_hist[min(seg_idx, step_time_hist.shape[0] - 1)]) if step_time_hist.size else seg_idx * sim_dt
        return _compute_fin_angles(refs, hist_in, c_a, fin_f, t_k, tau * sim_dt)

    def _build_fish_pose(render_idx: int) -> dict:
        seg_idx, tau = _render_segment(render_idx)
        fish_state = _interp_fish_state(fish_state_hist, seg_idx, tau)
        fin_angles = _get_fin_angles(seg_idx, tau)
        pos = fish_state[[PX, PY, PZ]]
        rot_body_to_world = quaternion_to_rotation_matrix(
            fish_state[Q0], fish_state[Q1], fish_state[Q2], fish_state[Q3]
        )

        right_fin_body = (_right_fin_rotation(float(fin_angles[A1]), float(fin_angles[A2])) @ fish_geom["right_fin_vertices"].T).T
        left_fin_body = (_left_fin_rotation(float(fin_angles[A3]), float(fin_angles[A4])) @ fish_geom["left_fin_vertices"].T).T
        tail_fin_body = (_tail_fin_rotation(float(fin_angles[A5])) @ fish_geom["tail_fin_vertices"].T).T

        right_fin_body = right_fin_body + fish_geom["right_attach"]
        left_fin_body = left_fin_body + fish_geom["left_attach"]
        tail_fin_body = tail_fin_body + fish_geom["tail_attach"]

        right_line_end_body = _rotation_x(float(fin_angles[A1])) @ np.array(
            [0.0, fish_geom["fin_offset_y"], 0.0], dtype=np.float64
        )
        left_line_end_body = _rotation_x(float(fin_angles[A3])) @ np.array(
            [0.0, -fish_geom["fin_offset_y"], 0.0], dtype=np.float64
        )
        tail_line_end_body = _rotation_y(float(fin_angles[A5])) @ np.array(
            [-fish_geom["tail_offset_x"], 0.0, 0.0], dtype=np.float64
        )

        return {
            "seg_idx": seg_idx,
            "tau": tau,
            "pos": pos,
            "body": _transform_points(fish_geom["body_vertices"], rot_body_to_world, pos),
            "right_fin": _transform_points(right_fin_body, rot_body_to_world, pos),
            "left_fin": _transform_points(left_fin_body, rot_body_to_world, pos),
            "tail_fin": _transform_points(tail_fin_body, rot_body_to_world, pos),
            "right_line": _transform_points(
                np.vstack([fish_geom["right_attach"], right_line_end_body + fish_geom["right_attach"]]),
                rot_body_to_world,
                pos,
            ),
            "left_line": _transform_points(
                np.vstack([fish_geom["left_attach"], left_line_end_body + fish_geom["left_attach"]]),
                rot_body_to_world,
                pos,
            ),
            "tail_line": _transform_points(
                np.vstack([fish_geom["tail_attach"], tail_line_end_body + fish_geom["tail_attach"]]),
                rot_body_to_world,
                pos,
            ),
        }

    fish_pose0 = _build_fish_pose(0)
    body_patch = Poly3DCollection(
        _body_face_vertices(fish_pose0["body"], fish_geom["body_faces"]),
        facecolor=(0.0, 0.7, 0.0, 0.90),
        edgecolor="k",
        linewidths=0.5,
    )
    fin_right_patch = Poly3DCollection(
        [fish_pose0["right_fin"]],
        facecolor=(1.0, 0.72, 0.22, 0.88),
        edgecolor="k",
        linewidths=0.5,
    )
    fin_left_patch = Poly3DCollection(
        [fish_pose0["left_fin"]],
        facecolor=(1.0, 0.92, 0.22, 0.88),
        edgecolor="k",
        linewidths=0.5,
    )
    fin_tail_patch = Poly3DCollection(
        [fish_pose0["tail_fin"]],
        facecolor=(0.22, 0.90, 0.35, 0.88),
        edgecolor="k",
        linewidths=0.5,
    )
    ax.add_collection3d(body_patch)
    ax.add_collection3d(fin_right_patch)
    ax.add_collection3d(fin_left_patch)
    ax.add_collection3d(fin_tail_patch)
    line1, = ax.plot(
        fish_pose0["right_line"][:, 0],
        fish_pose0["right_line"][:, 1],
        fish_pose0["right_line"][:, 2],
        color="k",
        linewidth=1.6,
    )
    line2, = ax.plot(
        fish_pose0["left_line"][:, 0],
        fish_pose0["left_line"][:, 1],
        fish_pose0["left_line"][:, 2],
        color="k",
        linewidth=1.6,
    )
    line3, = ax.plot(
        fish_pose0["tail_line"][:, 0],
        fish_pose0["tail_line"][:, 1],
        fish_pose0["tail_line"][:, 2],
        color="k",
        linewidth=1.6,
    )

    path_taken_plot, = ax.plot(
        valid_path[:1, 0],
        valid_path[:1, 1],
        valid_path[:1, 2],
        color=(0.0, 0.5, 0.0),
        linewidth=3.0,
        label="Actual Path",
    )
    local_tgt_plot, = ax.plot(
        [valid_path[0, 0]],
        [valid_path[0, 1]],
        [valid_path[0, 2]],
        "mx",
        markersize=9,
        markeredgewidth=2.0,
        visible=False,
        label="Local Target",
    )
    los_plot, = ax.plot(
        [valid_path[0, 0]],
        [valid_path[0, 1]],
        [valid_path[0, 2]],
        "cd",
        markersize=8,
        markeredgewidth=1.8,
        visible=False,
        label="LOS Target",
    )
    sensor_pc_plot = ax.scatter([], [], [], c="r", s=20, marker="o", label="Sensor Points")

    cone_x_base, cone_y_base, cone_z_base = _build_cone_mesh(
        float(sensor_params["range"]), float(sensor_params["fov_angle"])
    )
    cone_x, cone_y, cone_z = _transform_cone(
        cone_x_base, cone_y_base, cone_z_base, valid_path[0], initial_robot_vel
    )
    fov_cone_surface = ax.plot_surface(
        cone_x,
        cone_y,
        cone_z,
        color=(0.3, 0.8, 0.3),
        alpha=0.15,
        linewidth=0.4,
        edgecolor=(0.0, 0.6, 0.0),
    )

    ax.legend(loc="upper right")

    def _get_velocity(frame_idx: int) -> np.ndarray:
        seg_idx, tau = _render_segment(frame_idx)
        return _get_render_velocity(seg_idx, tau)

    def _get_blocked_pts(frame_idx: int) -> np.ndarray:
        seg_idx, _ = _render_segment(frame_idx)
        if seg_idx < len(blocked_pts_hist):
            pts = np.asarray(blocked_pts_hist[seg_idx], dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] == 3:
                return pts
        return np.zeros((0, 3), dtype=np.float64)

    def _get_target(target_hist: np.ndarray, frame_idx: int) -> np.ndarray:
        seg_idx, tau = _render_segment(frame_idx)
        if target_hist.ndim == 2 and target_hist.shape[0] > 0:
            if seg_idx >= target_hist.shape[0] - 1:
                return target_hist[min(seg_idx, target_hist.shape[0] - 1)]
            return (1.0 - tau) * target_hist[seg_idx] + tau * target_hist[seg_idx + 1]
        return np.full(3, np.nan, dtype=np.float64)

    def _update_dyn_obstacles(frame_idx: int):
        nonlocal dyn_surfaces
        if not dyn_surfaces:
            return
        seg_idx, tau = _render_segment(frame_idx)
        for surface in dyn_surfaces:
            surface.remove()
        dyn_surfaces = []
        for obs_idx in range(dyn_obs_hist.shape[1]):
            if seg_idx >= dyn_obs_hist.shape[0] - 1:
                obs_center = dyn_obs_hist[min(seg_idx, dyn_obs_hist.shape[0] - 1), obs_idx]
            else:
                obs_center = (1.0 - tau) * dyn_obs_hist[seg_idx, obs_idx] + tau * dyn_obs_hist[seg_idx + 1, obs_idx]
            x, y, z = _sphere_mesh(
                obs_center, float(dyn_obs_radii[obs_idx]), u, v
            )
            dyn_surfaces.append(
                ax.plot_surface(
                    x,
                    y,
                    z,
                    color=(0.2, 0.2, 0.9),
                    alpha=0.6,
                    linewidth=0.0,
                    edgecolor="none",
                )
            )
            trail_end = min(seg_idx + 1, dyn_obs_hist.shape[0])
            dyn_trails[obs_idx].set_data(
                dyn_obs_hist[:trail_end, obs_idx, 0],
                dyn_obs_hist[:trail_end, obs_idx, 1],
            )
            dyn_trails[obs_idx].set_3d_properties(dyn_obs_hist[:trail_end, obs_idx, 2])

    def _update(frame_idx: int):
        nonlocal fov_cone_surface

        fish_pose = _build_fish_pose(frame_idx)
        seg_idx = fish_pose["seg_idx"]
        pos = fish_pose["pos"]
        mode = mode_hist[seg_idx - 1] if seg_idx > 0 and seg_idx - 1 < len(mode_hist) else "GLOBAL_TRACKING"
        robot_color = (1.0, 0.0, 0.0) if mode == "LOCAL_AVOIDANCE" else (0.0, 0.7, 0.0)
        body_patch.set_verts(_body_face_vertices(fish_pose["body"], fish_geom["body_faces"]))
        body_patch.set_facecolor((*robot_color, 0.90))
        fin_right_patch.set_verts([fish_pose["right_fin"]])
        fin_left_patch.set_verts([fish_pose["left_fin"]])
        fin_tail_patch.set_verts([fish_pose["tail_fin"]])
        line1.set_data(fish_pose["right_line"][:, 0], fish_pose["right_line"][:, 1])
        line1.set_3d_properties(fish_pose["right_line"][:, 2])
        line2.set_data(fish_pose["left_line"][:, 0], fish_pose["left_line"][:, 1])
        line2.set_3d_properties(fish_pose["left_line"][:, 2])
        line3.set_data(fish_pose["tail_line"][:, 0], fish_pose["tail_line"][:, 1])
        line3.set_3d_properties(fish_pose["tail_line"][:, 2])
        path_prefix = valid_path[: min(seg_idx + 1, len(valid_path))]
        if path_prefix.size == 0:
            path_prefix = pos.reshape(1, 3)
        elif seg_idx < steps_executed:
            path_prefix = np.vstack([path_prefix, pos.reshape(1, 3)])
        path_taken_plot.set_data(path_prefix[:, 0], path_prefix[:, 1])
        path_taken_plot.set_3d_properties(path_prefix[:, 2])

        blocked_pts = _get_blocked_pts(frame_idx)
        if blocked_pts.size:
            sensor_pc_plot._offsets3d = (blocked_pts[:, 0], blocked_pts[:, 1], blocked_pts[:, 2])
        else:
            sensor_pc_plot._offsets3d = ([], [], [])

        los_target = _get_target(los_target_hist, frame_idx)
        if np.isfinite(los_target).all():
            los_plot.set_data([los_target[0]], [los_target[1]])
            los_plot.set_3d_properties([los_target[2]])
            los_plot.set_visible(True)
        else:
            los_plot.set_visible(False)

        local_target = _get_target(local_target_hist, frame_idx)
        if np.isfinite(local_target).all() and mode == "LOCAL_AVOIDANCE":
            local_tgt_plot.set_data([local_target[0]], [local_target[1]])
            local_tgt_plot.set_3d_properties([local_target[2]])
            local_tgt_plot.set_visible(True)
        else:
            local_tgt_plot.set_visible(False)

        cone_x, cone_y, cone_z = _transform_cone(
            cone_x_base, cone_y_base, cone_z_base, pos, _get_velocity(frame_idx)
        )
        fov_cone_surface.remove()
        fov_cone_surface = ax.plot_surface(
            cone_x,
            cone_y,
            cone_z,
            color=(0.3, 0.8, 0.3),
            alpha=0.15,
            linewidth=0.4,
            edgecolor=(0.0, 0.6, 0.0),
        )

        _update_dyn_obstacles(frame_idx)
        _set_3d_axis_scale(ax, axis_mins, axis_maxs)

        if seg_idx < len(title_hist):
            ax.set_title(title_hist[seg_idx])
        elif seg_idx > 0 and seg_idx - 1 < len(danger_dist_hist) and np.isfinite(danger_dist_hist[seg_idx - 1]):
            ax.set_title(f"Step {seg_idx}: {mode} - Visible Dist: {danger_dist_hist[seg_idx - 1]:.2f}")
        else:
            ax.set_title(f"Step {seg_idx}: {mode}")

        return (
            body_patch,
            fin_right_patch,
            fin_left_patch,
            fin_tail_patch,
            line1,
            line2,
            line3,
            path_taken_plot,
            local_tgt_plot,
            los_plot,
            sensor_pc_plot,
            goal_marker,
            *dyn_trails,
            *dyn_surfaces,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=render_frames,
        interval=15,
        blit=False,
        repeat=False,
    )
    fig._robot_anim = anim
    _LIVE_ANIMATIONS.append(anim)
    if save_path:
        _save_animation(anim, save_path, fps=fps)
    plt.tight_layout()

    if steps_executed > 0 and cmd_vel_des.shape[1] >= steps_executed and vel_act.shape[1] >= steps_executed:
        t_axis = np.arange(steps_executed, dtype=np.float64) * float(sensor_params.get("dt", 0.2))
        fig_vel, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor="white")
        fig_vel.suptitle("Velocity Tracking Performance")

        labels = ["v_x (m/s)", "v_y (m/s)", "v_z (m/s)"]
        colors = ["r", "g", "b"]
        for idx, ax_vel in enumerate(axes[:3]):
            ax_vel.plot(t_axis, cmd_vel_des[idx, :steps_executed], "r--", linewidth=1.5, label="Desired")
            ax_vel.plot(t_axis, vel_act[idx, :steps_executed], "b-", linewidth=1.0, label="Actual")
            ax_vel.set_ylabel(labels[idx])
            ax_vel.grid(True)
            if idx == 0:
                ax_vel.legend(loc="best")

        vel_error = cmd_vel_des[:, :steps_executed] - vel_act[:, :steps_executed]
        error_norm = np.sqrt(np.sum(vel_error**2, axis=0))
        axes[3].plot(t_axis, vel_error[0], "r-", linewidth=1.0, label="e_x")
        axes[3].plot(t_axis, vel_error[1], "g-", linewidth=1.0, label="e_y")
        axes[3].plot(t_axis, vel_error[2], "b-", linewidth=1.0, label="e_z")
        axes[3].plot(t_axis, error_norm, "k--", linewidth=1.5, label="|e|_total")
        axes[3].set_ylabel("Error (m/s)")
        axes[3].set_xlabel("Time (s)")
        axes[3].grid(True)
        axes[3].legend(loc="best")
        fig_vel.tight_layout()

    if steps_executed > 0:
        t_axis = np.arange(steps_executed, dtype=np.float64) * float(sensor_params.get("dt", 0.2))
        fig_att, axes_att = plt.subplots(4, 1, figsize=(11, 9), sharex=True, facecolor="white")
        fig_att.suptitle("Attitude Tracking Performance")
        psi_ref_plot = _unwrap_angle_series(psi_ref[:steps_executed])
        psi_act_plot = _unwrap_angle_series(psi_act[:steps_executed])

        axes_att[0].plot(t_axis, np.rad2deg(theta_ref[:steps_executed]), "r--", linewidth=1.5, label="theta_ref")
        axes_att[0].plot(t_axis, np.rad2deg(theta_act[:steps_executed]), "b-", linewidth=1.0, label="theta")
        axes_att[0].set_ylabel("Pitch (deg)")
        axes_att[0].grid(True)
        axes_att[0].legend(loc="best")

        axes_att[1].plot(t_axis, np.rad2deg(psi_ref_plot), "r--", linewidth=1.5, label="psi_ref")
        axes_att[1].plot(t_axis, np.rad2deg(psi_act_plot), "b-", linewidth=1.0, label="psi")
        axes_att[1].set_ylabel("Yaw (deg)")
        axes_att[1].grid(True)
        axes_att[1].legend(loc="best")

        axes_att[2].plot(t_axis, np.rad2deg(e_theta[:steps_executed]), "g-", linewidth=1.0, label="e_theta")
        axes_att[2].plot(t_axis, np.rad2deg(e_psi[:steps_executed]), "m-", linewidth=1.0, label="e_psi")
        axes_att[2].set_ylabel("Error (deg)")
        axes_att[2].grid(True)
        axes_att[2].legend(loc="best")

        axes_att[3].plot(t_axis, np.rad2deg(alpha5_ref[:steps_executed]), "c-", linewidth=1.0, label="alpha5_ref")
        axes_att[3].plot(t_axis, np.rad2deg(delta_ref[:steps_executed]), "k-", linewidth=1.0, label="delta_ref")
        axes_att[3].set_ylabel("Cmd (deg)")
        axes_att[3].set_xlabel("Time (s)")
        axes_att[3].grid(True)
        axes_att[3].legend(loc="best")
        fig_att.tight_layout()

    plt.show()


def plot_hybrid_comparison_result(
    rl_result: dict,
    mpc_result: dict,
    static_obs: list[dict],
    goal: np.ndarray,
    save_path: str | None = None,
    fps: int = 20,
) -> None:
    traj_global = np.asarray(rl_result["traj_global"]["xyz"], dtype=np.float64)
    rl_path = np.asarray(rl_result["path_hist"], dtype=np.float64)
    mpc_path = np.asarray(mpc_result["path_hist"], dtype=np.float64)
    rl_path = rl_path[np.isfinite(rl_path).all(axis=1)]
    mpc_path = mpc_path[np.isfinite(mpc_path).all(axis=1)]
    if rl_path.size == 0:
        rl_path = np.zeros((1, 3), dtype=np.float64)
    if mpc_path.size == 0:
        mpc_path = np.zeros((1, 3), dtype=np.float64)

    dyn_obs_hist = np.asarray(rl_result.get("dyn_obs_hist", []), dtype=np.float64)
    dyn_obs_radii = np.asarray(rl_result.get("dyn_obs_radii", []), dtype=np.float64)
    if dyn_obs_hist.ndim != 3:
        dyn_obs_hist = np.zeros((1, 0, 3), dtype=np.float64)
    if dyn_obs_radii.ndim != 1:
        dyn_obs_radii = np.zeros((0,), dtype=np.float64)

    bounds_points: list[np.ndarray] = [
        traj_global,
        rl_path,
        mpc_path,
        np.asarray(goal, dtype=np.float64).reshape(1, 3),
    ]
    if static_obs:
        static_bounds = []
        for obstacle in static_obs:
            center = np.asarray(obstacle["c"], dtype=np.float64).reshape(3)
            radius = float(obstacle["r"])
            static_bounds.extend([center - radius, center + radius])
        bounds_points.append(np.asarray(static_bounds, dtype=np.float64))
    if dyn_obs_hist.size > 0 and dyn_obs_hist.shape[1] > 0:
        for obs_idx in range(dyn_obs_hist.shape[1]):
            radius = float(dyn_obs_radii[obs_idx]) if obs_idx < dyn_obs_radii.size else 0.0
            bounds_points.append(dyn_obs_hist[:, obs_idx, :] - radius)
            bounds_points.append(dyn_obs_hist[:, obs_idx, :] + radius)

    axis_mins, axis_maxs = _axis_limits_from_points(bounds_points, min_span=2.0, pad_ratio=0.08)
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Hybrid Trajectory Planning Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    _set_3d_axis_scale(ax, axis_mins, axis_maxs)
    ax.set_autoscale_on(False)
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.view_init(elev=30.0, azim=-37.5)
    ax.grid(True)

    ax.plot(
        traj_global[:, 0],
        traj_global[:, 1],
        traj_global[:, 2],
        "k--",
        linewidth=1.4,
        label="Global Path",
    )

    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 12j]
    for obstacle in static_obs:
        x, y, z = _sphere_mesh(np.asarray(obstacle["c"], dtype=np.float64), float(obstacle["r"]), u, v)
        ax.plot_surface(
            x,
            y,
            z,
            color=(0.8, 0.2, 0.2),
            alpha=0.55,
            linewidth=0.0,
            edgecolor="none",
        )

    dyn_surfaces = []
    dyn_trails = []
    if dyn_obs_hist.shape[1] > 0:
        for obs_idx in range(dyn_obs_hist.shape[1]):
            x, y, z = _sphere_mesh(dyn_obs_hist[0, obs_idx], float(dyn_obs_radii[obs_idx]), u, v)
            dyn_surfaces.append(
                ax.plot_surface(
                    x,
                    y,
                    z,
                    color=(0.2, 0.2, 0.9),
                    alpha=0.55,
                    linewidth=0.0,
                    edgecolor="none",
                )
            )
            trail, = ax.plot(
                dyn_obs_hist[:1, obs_idx, 0],
                dyn_obs_hist[:1, obs_idx, 1],
                dyn_obs_hist[:1, obs_idx, 2],
                color=(0.2, 0.2, 0.9),
                linewidth=1.2,
                alpha=0.8,
            )
            dyn_trails.append(trail)

    ax.scatter(
        [goal[0]],
        [goal[1]],
        [goal[2]],
        c="y",
        s=180,
        marker="p",
        edgecolors="k",
        label="Goal",
    )

    rl_robot = ax.scatter(
        [rl_path[0, 0]],
        [rl_path[0, 1]],
        [rl_path[0, 2]],
        c=[(0.90, 0.25, 0.20)],
        s=120,
        marker="o",
        edgecolors="k",
        label="RL Robot",
    )
    mpc_robot = ax.scatter(
        [mpc_path[0, 0]],
        [mpc_path[0, 1]],
        [mpc_path[0, 2]],
        c=[(0.15, 0.40, 0.90)],
        s=120,
        marker="^",
        edgecolors="k",
        label="MPC Robot",
    )
    rl_path_plot, = ax.plot(
        rl_path[:1, 0],
        rl_path[:1, 1],
        rl_path[:1, 2],
        color=(0.95, 0.40, 0.15),
        linewidth=2.8,
        label="RL Path",
    )
    mpc_path_plot, = ax.plot(
        mpc_path[:1, 0],
        mpc_path[:1, 1],
        mpc_path[:1, 2],
        color=(0.10, 0.45, 0.95),
        linewidth=2.6,
        label="MPC Path",
    )
    ax.legend(loc="upper right")

    rl_mode_hist = list(rl_result.get("settings", {}).get("mode", []))
    mpc_mode_hist = list(mpc_result.get("settings", {}).get("mode", []))

    def _get_pos(path_hist: np.ndarray, frame_idx: int) -> np.ndarray:
        return path_hist[min(frame_idx, len(path_hist) - 1)]

    def _get_mode(mode_hist: list[str], frame_idx: int) -> str:
        if not mode_hist:
            return "GLOBAL_TRACKING"
        if frame_idx <= 0:
            return "GLOBAL_TRACKING"
        return mode_hist[min(frame_idx - 1, len(mode_hist) - 1)]

    def _update_dyn_obstacles(frame_idx: int):
        nonlocal dyn_surfaces
        if not dyn_surfaces:
            return
        hist_idx = min(frame_idx, dyn_obs_hist.shape[0] - 1)
        for surface in dyn_surfaces:
            surface.remove()
        dyn_surfaces = []
        for obs_idx in range(dyn_obs_hist.shape[1]):
            x, y, z = _sphere_mesh(dyn_obs_hist[hist_idx, obs_idx], float(dyn_obs_radii[obs_idx]), u, v)
            dyn_surfaces.append(
                ax.plot_surface(
                    x,
                    y,
                    z,
                    color=(0.2, 0.2, 0.9),
                    alpha=0.55,
                    linewidth=0.0,
                    edgecolor="none",
                )
            )
            dyn_trails[obs_idx].set_data(
                dyn_obs_hist[: hist_idx + 1, obs_idx, 0],
                dyn_obs_hist[: hist_idx + 1, obs_idx, 1],
            )
            dyn_trails[obs_idx].set_3d_properties(dyn_obs_hist[: hist_idx + 1, obs_idx, 2])

    def _update(frame_idx: int):
        rl_pos = _get_pos(rl_path, frame_idx)
        mpc_pos = _get_pos(mpc_path, frame_idx)

        rl_robot._offsets3d = ([rl_pos[0]], [rl_pos[1]], [rl_pos[2]])
        mpc_robot._offsets3d = ([mpc_pos[0]], [mpc_pos[1]], [mpc_pos[2]])
        rl_path_plot.set_data(rl_path[: min(frame_idx + 1, len(rl_path)), 0], rl_path[: min(frame_idx + 1, len(rl_path)), 1])
        rl_path_plot.set_3d_properties(rl_path[: min(frame_idx + 1, len(rl_path)), 2])
        mpc_path_plot.set_data(mpc_path[: min(frame_idx + 1, len(mpc_path)), 0], mpc_path[: min(frame_idx + 1, len(mpc_path)), 1])
        mpc_path_plot.set_3d_properties(mpc_path[: min(frame_idx + 1, len(mpc_path)), 2])
        _update_dyn_obstacles(frame_idx)
        _set_3d_axis_scale(ax, axis_mins, axis_maxs)

        rl_mode = _get_mode(rl_mode_hist, frame_idx)
        mpc_mode = _get_mode(mpc_mode_hist, frame_idx)
        ax.set_title(f"Step {frame_idx} | RL: {rl_mode} | MPC: {mpc_mode}")
        return rl_robot, mpc_robot, rl_path_plot, mpc_path_plot, *dyn_trails, *dyn_surfaces

    frames = max(len(rl_path), len(mpc_path), dyn_obs_hist.shape[0])
    anim = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=15,
        blit=False,
        repeat=False,
    )
    fig._robot_anim = anim
    _LIVE_ANIMATIONS.append(anim)
    if save_path:
        _save_animation(anim, save_path, fps=fps)
    plt.tight_layout()

    plt.show()
