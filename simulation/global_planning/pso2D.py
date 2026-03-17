import numpy as np

def pso2D(objFun, lb, ub, opts):
    """Particle Swarm Optimization in 2D."""
    M = opts['psoM']
    T = opts['psoT']

    # Initialize positions
    width = np.array(ub) - np.array(lb)
    x = np.array(lb) + width * np.random.rand(M, 2)

    # Initialize velocities
    v = 0.1 * width * np.random.randn(M, 2)

    # Best tracking
    pbest = x.copy()
    pbestJ = np.full(M, np.inf)

    # Initial evaluation
    for i in range(M):
        pbestJ[i] = objFun(x[i, :])
    
    idx_min = np.argmin(pbestJ)
    bestJ = pbestJ[idx_min]
    bestX = pbest[idx_min, :].copy()

    w = opts['w']
    c1 = opts['c1']
    c2 = opts['c2']

    for k in range(T):
        r1 = np.random.rand(M, 2)
        r2 = np.random.rand(M, 2)

        # Update velocity and position
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (bestX - x)
        x = x + v

        # Clamp bounds
        x[:, 0] = np.clip(x[:, 0], lb[0], ub[0])
        x[:, 1] = np.clip(x[:, 1], lb[1], ub[1])

        # Evaluate
        for i in range(M):
            Ji = objFun(x[i, :])
            if Ji < pbestJ[i]:
                pbestJ[i] = Ji
                pbest[i, :] = x[i, :]
                if Ji < bestJ:
                    bestJ = Ji
                    bestX = x[i, :]
    
    return bestX, bestJ
