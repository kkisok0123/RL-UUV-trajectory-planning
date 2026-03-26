import numpy as np

def bernstein3(tau):
    """Cubic Bernstein basis."""
    t = tau.reshape(-1, 1) # Ensure column vector
    B0 = (1 - t)**3
    B1 = 3 * (1 - t)**2 * t
    B2 = 3 * (1 - t) * t**2
    B3 = t**3
    return np.hstack([B0, B1, B2, B3])
