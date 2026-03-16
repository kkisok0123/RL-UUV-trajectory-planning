# dynamics.py
import numpy as np

def dynamics_step(state, action, dt):
    """
    state: [x, y, z, vx, vy, vz]
    action: [Fx, Fy, Fz]
    """
    x, y, z, vx, vy, vz = state
    Fx, Fy, Fz = action

    m = 10.0
    drag = 0.8

    ax = (Fx - drag * vx) / m
    ay = (Fy - drag * vy) / m
    az = (Fz - drag * vz) / m

    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    vz_next = vz + az * dt

    x_next = x + vx_next * dt
    y_next = y + vy_next * dt
    z_next = z + vz_next * dt

    next_state = np.array([x_next, y_next, z_next, vx_next, vy_next, vz_next], dtype=np.float32)
    return next_state