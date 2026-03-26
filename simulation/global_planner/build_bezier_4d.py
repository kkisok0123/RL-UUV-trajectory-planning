import numpy as np

try:
    from .bernstein3 import bernstein3
except ImportError:
    from bernstein3 import bernstein3

def buildBezier4D(Fs, Fg, t0, t1, t2, t3, N):
    """Builds the 4D Bezier curve [x, y, z, t]."""
    # Control points in 4D: [x, y, z, t]
    P0 = np.hstack([Fs['p'], t0])
    P3 = np.hstack([Fg['p'], t3])

    # Paper Eq. (15): Compute P1, P2 from velocities and times
    p1_xyz = Fs['p'] + Fs['v'] * (t1 - t0)
    p2_xyz = Fg['p'] + Fg['v'] * (t3 - t2)

    P1 = np.hstack([p1_xyz, t1])
    P2 = np.hstack([p2_xyz, t2])

    P4 = np.vstack([P0, P1, P2, P3]) # 4x4 matrix

    tau = np.linspace(0, 1, N)
    B = bernstein3(tau) # Nx4

    X4 = B @ P4 # Matrix multiplication: Nx4
    return P4, tau, X4
