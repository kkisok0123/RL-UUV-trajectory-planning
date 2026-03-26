import numpy as np

try:
    from .build_bezier_4d import buildBezier4D
except ImportError:
    from build_bezier_4d import buildBezier4D

def globalBezierObjective(tt, Fs, Fg, t0, t3, opts, obstacles):
    """Objective function for PSO."""
    t1, t2 = tt[0], tt[1]

    # Enforce bounds and ordering
    penOrder = 0
    if (t1 < t0) or (t2 < t0) or (t1 > t3) or (t2 > t3) or (t1 > t2):
        penOrder = opts['Mk_order'] * (
            max(0, t0 - t1)**2 + max(0, t0 - t2)**2 +
            max(0, t1 - t3)**2 + max(0, t2 - t3)**2 +
            max(0, t1 - t2)**2
        )

    # Build curve
    _, _, X4 = buildBezier4D(Fs, Fg, t0, t1, t2, t3, opts['N'])
    xyz = X4[:, 0:3]
    t = X4[:, 3]

    # Time monotonicity check
    dt = np.diff(t)
    if np.any(dt <= 1e-9):
        return 1e12 + penOrder

    # Path length S
    dxyz = np.diff(xyz, axis=0)
    segL = np.sqrt(np.sum(dxyz**2, axis=1))
    S = np.sum(segL)

    # Velocity / Acceleration
    v = dxyz / dt.reshape(-1, 1)
    speed = np.sqrt(np.sum(v**2, axis=1))

    dv = np.diff(v, axis=0)
    dt_mid = (dt[:-1] + dt[1:]) / 2
    a = dv / dt_mid.reshape(-1, 1)
    acc = np.sqrt(np.sum(a**2, axis=1))

    # Curvature
    v_mid = v[:-1, :]
    vnorm = np.sqrt(np.sum(v_mid**2, axis=1))
    # Cross product in 3D
    cross_va = np.cross(v_mid, a)
    kappa = np.sqrt(np.sum(cross_va**2, axis=1)) / np.maximum(vnorm**3, 1e-9)

    # Smoother Constraint Penalties
    penSpeed = np.sum(np.maximum(0, speed - opts['vmax'])**2)
    penAcc = np.sum(np.maximum(0, acc - opts['amax'])**2)
    # Integral of acceleration squared (Smoothness)
    penSmooth = np.sum(acc**2) * (t3 - t0) / len(acc) 
    penKappa = np.sum(np.maximum(0, kappa - opts['kappamax'])**2)

    # Obstacle Penalties (Soft Quadratic Barrier)
    penObs = 0
    d_margin = 1.0 # Distance at which penalty starts to increase
    for obs in obstacles:
        c = obs['c']
        r = obs['r'] + opts['clearance']
        # Compute distances
        dist = np.sqrt(np.sum((xyz - c)**2, axis=1))
        gap = dist - r 

        # Collision penalty
        penObs += 1e5 * np.sum(np.maximum(0, -gap)**2) 
        # Softer proximity penalty (quadratic when within d_margin)
        penObs += np.sum(np.maximum(0, d_margin - gap)**2)

    J = (S + 
         opts['Mk_speed'] * penSpeed + 
         opts['Mk_acc'] * penAcc + 
         1.0 * penSmooth + 
         opts['Mk_kappa'] * penKappa + 
         opts['Mk_obs'] * penObs + 
         penOrder)
    
    return J
