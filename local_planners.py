import numpy as np

def get_nearest_obstacle(pos, obstacles):
    d_min = float('inf')
    nearest = None
    for obs in obstacles:
        d = np.linalg.norm(pos - obs['c']) - obs['r']
        if d < d_min:
            d_min = d
            nearest = obs
    return nearest, d_min

def find_local_target(traj, obs, search_dist, current_pos):
    """Scans global path for a re-entry point that is AHEAD of the current position."""
    xyz = traj['xyz']
    
    # First, find the closest point on the trajectory to current position
    dists_to_traj = np.linalg.norm(xyz - current_pos, axis=1)
    closest_idx = np.argmin(dists_to_traj)
    
    # Search for a re-entry point starting from ahead of closest_idx
    for i in range(closest_idx, len(xyz)):
        pt = xyz[i]
        d_surf = np.linalg.norm(pt - obs['c']) - obs['r']
        if d_surf > search_dist:
            return pt
            
    return xyz[-1] # Default to end

def local_planner_mock_ippo(curr_pos, obs, target_pos, dt):
    """Simulates IPPO output using balanced Potential Fields."""
    # 1. Attraction to target
    vec_to_goal = target_pos - curr_pos
    dist_goal = np.linalg.norm(vec_to_goal)
    if dist_goal > 1e-3:
        # Linear attraction, stronger when far
        F_att = 1.5 * (vec_to_goal / dist_goal)
    else:
        F_att = np.zeros(3)
        
    # 2. Repulsion from obstacle (clamped strength)
    vec_from_obs = curr_pos - obs['c']
    dist_center = np.linalg.norm(vec_from_obs)
    dist_surface = dist_center - obs['r']
    
    influence_range = 2.0
    if dist_surface < influence_range and dist_surface > 0.01:
        # Clamped inverse repulsion
        strength = min(5.0, 1.0 / (dist_surface**2))
        F_rep = strength * (vec_from_obs / dist_center)
    elif dist_surface <= 0.01:
        # Very close or inside: strong push outward
        F_rep = 10.0 * (vec_from_obs / max(dist_center, 0.1))
    else:
        F_rep = np.zeros(3)
        
    # 3. Combine with balanced weights
    vel_cmd = F_att + 0.5 * F_rep
    
    # Cap speed
    v_max = 1.5
    speed = np.linalg.norm(vel_cmd)
    if speed > v_max:
        vel_cmd = (vel_cmd / speed) * v_max
        
    next_pos = curr_pos + vel_cmd * dt
    return next_pos
