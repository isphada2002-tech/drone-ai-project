import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import time
import os
import pyvista as pv

# 🔗 Import the continuous 3D Double DQN architecture
from ddqn_agent import DoubleDQNAgent

print("--> [1] Program started successfully...")

# Real-world terrain dataset configuration
tif_filename = 'output_SRTMGL1.tif'
print(f"--> [2] Looking for the file: '{tif_filename}'...")

try:
    img = Image.open(tif_filename)
    print("--> [3] File found and opened by Pillow. Extracting array...")
    Z_raw = np.array(img, dtype=float)
    print(f"--> [4] Array extracted. Original dimensions: {Z_raw.shape}")

    # 🛡️ Safe Downsampling for memory handling
    downsample_factor = 20 
    Z = Z_raw[::downsample_factor, ::downsample_factor]
    print(f"--> [5] Data downsampled for safety. New grid size: {Z.shape}")

    Z[Z < -100] = 0
    Z_flip = np.flipud(Z)
except Exception as e:
    print(f"❌ Error during file processing: {e}")
    sys.exit()

print("--> [6] Generating 3D coordinates for the real terrain...")
resolution = 30.0 * downsample_factor
x_real = np.arange(0, Z_flip.shape[1] * resolution, resolution)
y_real = np.arange(0, Z_flip.shape[0] * resolution, resolution)
X, Y = np.meshgrid(x_real, y_real)

print("--> [7] Injecting data into PyVista Structured Grid...")
grid = pv.StructuredGrid(X, Y, Z_flip)
grid.points[:, 2] = Z_flip.ravel()
grid['elevation'] = Z_flip.ravel()

print("--> [8] Initializing Plotter engine...")
plotter = pv.Plotter(window_size=[1024, 768])
plotter.add_mesh(grid, cmap='terrain', smooth_shading=True, show_scalar_bar=True)
plotter.set_scale(zscale=10.0)
plotter.camera.elevation = 35
plotter.camera.azimuth = -45

# 🤖 Initialize 3D Double DQN agent: 6 actions (3D control), 3 sensor inputs
ai_agent = DoubleDQNAgent(actions_count=6, input_dim=3, hidden_dim=16)
ai_agent.load_weights()

# 🛸 Flight Path and Environment Simulation Loop
# =================================================================
print("--> [9] Simulating flight path with True 3D Control and Vertical Turbulence...")

num_steps = 150
x_true = np.linspace(x_real[0], x_real[-1], num_steps)
y_true = np.linspace(y_real[0], y_real[-1], num_steps)

# Define baseline operational cruise altitude profile
safe_altitude = np.max(Z_flip) + 300
z_true = np.ones(num_steps) * safe_altitude
true_points = np.column_stack((x_true, y_true, z_true))

episode_rewards = []

for episode in range(1000):
    should_render = (episode == 999)
    ai_corrected_points = []
    
    # Reset all 3D drifts at the beginning of each episode
    current_drift_x = 0
    current_drift_y = 0
    current_drift_z = 0 
    
    gps_loss_step = 30
    total_episode_reward = 0

    if should_render:
        plotter.add_lines(true_points, color='green', width=5, label='Planned Route')
        drift_line_actor = None
        drone_actor = plotter.add_mesh(pv.Sphere(radius=400, center=true_points[0]), color='yellow')
        plotter.show(interactive_update=True)
        print(f"\n--> [10] SHOWING FINAL LIVE SIMULATION FOR EPISODE {episode + 1}/1000...")
    elif (episode + 1) % 10 == 0:
        print(f"--> Training in background... Episode {episode + 1}/1000 complete.")

    for step in range(num_steps):
        tx, ty, tz = true_points[step]
        reward = 0

        # Calculate live spatial coordinates including 3D tracking offsets
        actual_x = tx + current_drift_x
        actual_y = ty + current_drift_y
        actual_z = tz + current_drift_z

        # Extract local ground elevation index underneath the drone
        idx_x = int(min(max(actual_x / resolution, 0), Z_flip.shape[1] - 1))
        idx_y = int(min(max(actual_y / resolution, 0), Z_flip.shape[0] - 1))
        terrain_height = Z_flip[idx_y, idx_x]
        
        # 📡 Virtual LiDAR clearance sensor evaluation
        current_clearance = actual_z - terrain_height

        if step >= gps_loss_step:
            # Active environmental simulation: Horizontal wind drift + severe vertical turbulence
            current_drift_x += np.random.normal(0, 180)
            current_drift_y += np.random.normal(0, 180)
            current_drift_z += np.random.normal(0, 60)  # Vertical wind effect

            # Save states for update parameters mapping
            state_x, state_y, state_clearance = current_drift_x, current_drift_y, current_clearance
            
            # Predict intelligent 3D maneuver choice via DDQN brain
            action = ai_agent.choose_action(current_drift_x, current_drift_y, current_clearance)

            correction_step = 120
            if action == 0:    current_drift_x -= correction_step # Move Forward
            elif action == 1:  current_drift_x += correction_step # Move Backward
            elif action == 2:  current_drift_y -= correction_step # Move Left
            elif action == 3:  current_drift_y += correction_step # Move Right
            elif action == 4:  current_drift_z += correction_step # Mechanical Climb ⬆
            elif action == 5:  current_drift_z -= correction_step # Mechanical Descent ⬇

            # Compute horizontal clearance offset metric
            current_dist_err = np.sqrt(current_drift_x**2 + current_drift_y**2)

            # Move to next temporal frame with standard noise
            next_drift_x = current_drift_x + np.random.normal(0, 10)
            next_drift_y = current_drift_y + np.random.normal(0, 10)
            next_drift_z = current_drift_z + np.random.normal(0, 5)
            
            next_dist_err = np.sqrt(next_drift_x**2 + next_drift_y**2)
            
            # Future sensor simulation step for temporal difference target
            next_idx_x = int(min(max((tx + next_drift_x) / resolution, 0), Z_flip.shape[1] - 1))
            next_idx_y = int(min(max((ty + next_drift_y) / resolution, 0), Z_flip.shape[0] - 1))
            next_clearance = (tz + next_drift_z) - Z_flip[next_idx_y, next_idx_x]

            # 🛠 Compound 3D Reward Shaping Function
            reward = -next_dist_err / 10.0

            if next_dist_err < current_dist_err:
                reward += 25
            else:
                reward -= 25.0
            if next_dist_err < 60:
                reward += 100
                
            # 🛑 CRITICAL CRASH AVOIDANCE CONSTRAINT LOGIC
            if next_clearance < 150.0:  # Severe mountain proximity limit
                reward -= 600.0         # Heavy mathematical penalty applied
            elif next_clearance >= 300.0:
                reward += 40.0          # Safe cruise bonus reward

            # Execute backpropagation training live
            ai_agent.store_transition(state_x, state_y, state_clearance, action, reward, next_drift_x, next_drift_y, next_clearance)
            ai_agent.replay_and_learn()

        total_episode_reward += reward
        ai_corrected_points.append([actual_x, actual_y, actual_z])

        # 3D Visual Mesh rendering engine update
        if should_render:
            if len(ai_corrected_points) > 1:
                if drift_line_actor: plotter.remove_actor(drift_line_actor)
                line_mesh = pv.lines_from_points(np.array(ai_corrected_points))
                drift_line_actor = plotter.add_mesh(line_mesh, color='red', line_width=4)

            drone_actor.mapper.dataset.copy_from(pv.Sphere(radius=400, center=[actual_x, actual_y, actual_z]))
            plotter.camera.azimuth += 0.25
            plotter.update()
            time.sleep(0.02)

    # Exploration rate decay step
    ai_agent.epsilon = max(0.01, ai_agent.epsilon * 0.9992)
    episode_rewards.append(total_episode_reward)

print("\n--> [11] 3D DDQN Training Complete! Saving weights...")
ai_agent.save_weights()

if 'plotter' in locals():
    plotter.show()

# Render performance learning graph via Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, color='crimson', linewidth=2, label='Total Reward')
plt.title('3D Double DQN Learning Curve with 6-DOF Action & LiDAR Clearance', fontsize=12, fontweight='bold')
plt.xlabel('Episodes')
plt.ylabel('Accumulated Reward')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
smoothed_rewards = np.convolve(episode_rewards, np.ones(15)/15, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, color='crimson', alpha=0.3, label='Raw Episode Reward')
plt.plot(smoothed_rewards, color='blue', linewidth=2.5, label='15-Episode Moving Average (Stabilized)')
plt.title('3D Double DQN Convergence Curve (Enriched with Replay Buffer Memory)', fontsize=11, fontweight='bold')
plt.xlabel('Episodes')
plt.ylabel('Accumulated Reward')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
