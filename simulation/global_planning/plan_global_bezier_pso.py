import numpy as np

try:
    from .build_bezier_4d import buildBezier4D
    from .global_bezier_objective import globalBezierObjective
    from .pso2D import pso2D
except ImportError:
    from build_bezier_4d import buildBezier4D
    from global_bezier_objective import globalBezierObjective
    from pso2D import pso2D

def planGlobalBezierPSO(Fs, Fg, opts, obstacles):
    """Main wrapper for global planning."""
    # Set defaults if keys missing
    defaults = {
        't0': 0.0, 't3': 20.0, 'N': 200, 'psoM': 50, 'psoT': 50,
        'w': 0.7, 'c1': 1.6, 'c2': 1.6,
        'vmax': 2.0, 'amax': 1.0, 'kappamax': 2.0, 'clearance': 0.5,
        'Mk_speed': 1e5, 'Mk_acc': 1e5, 'Mk_kappa': 1e5, 'Mk_obs': 1, 'Mk_order': 1e6
    }
    for k, v in defaults.items():
        if k not in opts:
            opts[k] = v
            
    t0 = opts['t0']
    t3 = opts['t3']
    
    lb = [t0, t0]
    ub = [t3, t3]
    
    # Lambda for objective
    func = lambda tt: globalBezierObjective(tt, Fs, Fg, t0, t3, opts, obstacles)
    
    bestTT, bestCost = pso2D(func, lb, ub, opts)
    
    P4, tau, X4 = buildBezier4D(Fs, Fg, t0, bestTT[0], bestTT[1], t3, opts['N'])
    
    return {
        'cost': bestCost,
        't0': t0, 't1': bestTT[0], 't2': bestTT[1], 't3': t3,
        'P4': P4, 'tau': tau, 't': X4[:, 3], 'xyz': X4[:, 0:3]
    }
