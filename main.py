import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from planGlobalBezierPSO import planGlobalBezierPSO
from local_planners import get_nearest_obstacle, find_local_target, local_planner_mock_ippo

def main():
    # Setup Start/Goal
    Fs = {'p': np.array([0.0, 0.0, 2.0]), 'v': np.array([0.5, 0.5, 0.0])}
    Fg = {'p': np.array([12.0, 12.0, 8.0]), 'v': np.array([0.0, 0.0, 0.0])}
    
    # Static obstacles
    static_obstacles = [
        {'c': np.array([3.0, 3.0, 3.0]), 'r': 1.0},
        {'c': np.array([6.0, 8.0, 5.0]), 'r': 1.5},
        {'c': np.array([9.0, 4.0, 6.0]), 'r': 1.2},
    ]
    
    opts = {'t3': 50.0, 'vmax': 1.5, 'psoM': 100, 'psoT': 100}
    
    print("Planning Global Trajectory...")
    global_traj = planGlobalBezierPSO(Fs, Fg, opts, static_obstacles)
    print(f"Global Plan Done. Cost: {global_traj['cost']:.4f}")
    
    # Simulation Config
    dt = 0.05
    total_time = global_traj['t3']
    current_pos = Fs['p'].copy()
    
    # Ground Grid for Visualization
    grid_res = 20
    gx, gy = np.meshgrid(np.linspace(-2, 14, grid_res), np.linspace(-2, 14, grid_res))
    gz = np.zeros_like(gx)

    dynamic_obstacles = [
        {'c': np.array([5.0, 5.0, 5.0]), 'r': 1.0, 'v': np.array([-0.2, 0.1, 0.0]), 'color': '#FF9933'},
        {'c': np.array([8.0, 9.0, 7.0]), 'r': 0.8, 'v': np.array([0.1, -0.2, 0.1]), 'color': '#9933FF'}
    ]
    
    safe_margin = 0.8
    history_pos = [current_pos.copy()]
    
    # Setup Plot
    plt.close('all')
    plt.ion()
    fig = plt.figure(figsize=(14, 10), facecolor='#121212')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#121212')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    # Simulation Loop
    times = np.arange(0, total_time, dt)
    
    # Mode Hysteresis
    mode = "GLOBAL TRACKING"
    d_enter = safe_margin + 0.2
    d_exit = d_enter + 0.4
    
    for t in times:
        # A. Update Dynamic Obstacles
        for obs in dynamic_obstacles:
            obs['c'] += obs['v'] * dt
        
        # B. Control Logic
        all_obstacles = static_obstacles + dynamic_obstacles
        nearest, dist_min = get_nearest_obstacle(current_pos, all_obstacles)
        
        # Hysteresis Logic
        if mode == "GLOBAL TRACKING":
            if dist_min < d_enter:
                mode = "LOCAL AVOIDANCE"
        else: # LOCAL AVOIDANCE
            if dist_min > d_exit:
                mode = "GLOBAL TRACKING"

        if mode == "LOCAL AVOIDANCE":
            local_target = find_local_target(global_traj, nearest, d_exit + 0.5, current_pos)
            next_pos = local_planner_mock_ippo(current_pos, nearest, local_target, dt)
        else: # GLOBAL TRACKING
            # Lookahead index
            idx = (np.abs(global_traj['t'] - (t + 0.2))).argmin()
            target_pt = global_traj['xyz'][idx]
            vec = target_pt - current_pos
            dist_vec = np.linalg.norm(vec)
            if dist_vec > 1e-3:
                # Use slightly higher tracking speed if far away
                speed_gain = 1.2 if dist_vec > 0.5 else 1.0
                step_size = min(dist_vec, opts['vmax'] * dt * speed_gain)
                next_pos = current_pos + (vec / dist_vec) * step_size
            else:
                next_pos = current_pos

        current_pos = next_pos
        history_pos.append(current_pos.copy())
        
        # C. Enhanced Visualization
        if int(t/dt) % 3 == 0:
            ax.cla()
            ax.set_facecolor('#121212')
            ax.set_xlim(-2, 14); ax.set_ylim(-2, 14); ax.set_zlim(0, 10)
            ax.set_xlabel('X', color='gray'); ax.set_ylabel('Y', color='gray'); ax.set_zlabel('Z', color='gray')
            ax.tick_params(colors='gray')
            
            # Ground Grid
            ax.plot_wireframe(gx, gy, gz, color='cyan', alpha=0.05, linewidth=0.5)
            
            # Shadows
            hist_arr = np.array(history_pos)
            ax.plot(hist_arr[:,0], hist_arr[:,1], 0, 'white', alpha=0.15, linewidth=1)
            ax.plot([current_pos[0], current_pos[0]], [current_pos[1], current_pos[1]], [0, current_pos[2]], 
                    color='white', alpha=0.1, linestyle=':')
            ax.scatter(current_pos[0], current_pos[1], 0, color='white', s=10, alpha=0.2)

            # Trajectories
            ax.plot(global_traj['xyz'][:,0], global_traj['xyz'][:,1], global_traj['xyz'][:,2], 
                    color='#00FFCC', alpha=0.15, linestyle='--', linewidth=1)
            ax.plot(hist_arr[:,0], hist_arr[:,1], hist_arr[:,2], 
                    color='#FF3366', linewidth=2, alpha=0.8, label='Robot Path')
            
            # Robot
            ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='#FF3366', s=50, edgecolors='white', zorder=10)

            # Obstacles
            u, v_sph = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
            for obs in all_obstacles:
                color = obs.get('color', '#3399FF')
                x_sph = obs['c'][0] + obs['r'] * np.cos(u) * np.sin(v_sph)
                y_sph = obs['c'][1] + obs['r'] * np.sin(u) * np.sin(v_sph)
                z_sph = obs['c'][2] + obs['r'] * np.cos(v_sph)
                ax.plot_surface(x_sph, y_sph, z_sph, color=color, alpha=0.3, shade=True)
                
                # Obstacle Shadow
                ax.plot(obs['c'][0] + obs['r']*np.cos(np.linspace(0, 2*np.pi, 30)), 
                        obs['c'][1] + obs['r']*np.sin(np.linspace(0, 2*np.pi, 30)), 
                        0, color='white', alpha=0.05)

            # Info HUD
            dist_to_goal = np.linalg.norm(current_pos - Fg['p'])
            info_text = (f"Time: {t:4.1f}s\n"
                         f"Mode: {mode}\n"
                         f"Dist to Goal: {dist_to_goal:4.2f}m\n"
                         f"Obstacles: {len(all_obstacles)}")
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, color='white', va='top',
                      fontsize=9, bbox=dict(facecolor='black', alpha=0.6, edgecolor='#00FFCC'))

            ax.set_title("Robust 3D Trajectory Avoiding Navigation", color='white', pad=15)
            
            plt.draw()
            plt.pause(0.001)
            
        if dist_to_goal < 0.6:
            print("Goal Reached!")
            break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
