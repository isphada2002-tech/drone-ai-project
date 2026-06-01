import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import time
import pickle
import os

print("--> [1] Program started successfully...")

# Check if PyVista and VTK can load without crashing IDLE
try:
    import pyvista as pv
    print("--> [2] PyVista and VTK libraries loaded fine!")
except Exception as e:
    print(f"❌ Error during PyVista import: {e}")
    sys.exit()

# File path configuration
tif_filename = 'output_SRTMGL1.tif'
print(f"--> [3] Looking for the file: '{tif_filename}'...")

try:
    img = Image.open(tif_filename)
    print("--> [4] File found and opened by Pillow. Extracting array...")
    Z_raw = np.array(img, dtype=float)
    print(f"--> [5] Array extracted. Original dimensions: {Z_raw.shape}")
    
    # 🛡️ SAFE DOWNSAMPLING: Keeping it at 20 for Mac system safety
    downsample_factor = 20 
    Z = Z_raw[::downsample_factor, ::downsample_factor]
    print(f"--> [6] Data downsampled for safety. New grid size: {Z.shape}")
    
    Z[Z < -100] = 0
    Z_flip = np.flipud(Z)
except Exception as e:
    print(f"❌ Error during file processing: {e}")
    print("Please check if the file is in the correct folder and named properly!")
    sys.exit()

print("--> [7] Generating 3D coordinates for the real terrain...")
resolution = 30.0 * downsample_factor
x_real = np.arange(0, Z_flip.shape[1] * resolution, resolution)
y_real = np.arange(0, Z_flip.shape[0] * resolution, resolution)
X, Y = np.meshgrid(x_real, y_real)

print("--> [8] Injecting data into PyVista Structured Grid...")
grid = pv.StructuredGrid(X, Y, Z_flip)
grid.points[:, 2] = Z_flip.ravel()
grid['elevation'] = Z_flip.ravel()

print("--> [9] Initializing Plotter engine with Vertical Exaggeration...")
plotter = pv.Plotter(window_size=[1024, 768])

# Adding the real terrain mesh with beautiful shading and colors
plotter.add_mesh(grid, 
                 cmap='terrain', 
                 smooth_shading=True, 
                 show_scalar_bar=True,
                 scalar_bar_args={'title': 'Real Elevation (m)'})

plotter.set_scale(zscale=10.0)
plotter.add_text("Tangier-Tetouan Autonomous RL Navigation (Z-Scale x10)", font_size=14, color='black')

plotter.camera.elevation = 35
plotter.camera.azimuth = -45
plotter.show_bounds()
plotter.add_axes()

# =================================================================
# 🧠 Q-LEARNING AI AGENT IMPLEMENTATION
# =================================================================
class QLearningAgent:
    def __init__(self, actions_count=5):
        # Actions: 0=Left, 1=Right, 2=Down, 3=Up, 4=No Action
        self.actions_count = actions_count
        self.lr = 0.2        # Learning Rate
        self.discount = 0.9  # Gamma discount factor
        self.epsilon = 0.3   # Exploration rate for epsilon-greedy strategy
        
        # Initialize a dynamic Q-Table using a dictionary to map states to action rewards
        self.q_table = {}

    def _get_state_key(self, drift_x, drift_y):
        # Discretize the continuous spatial drift to create categorical states
        state_x = int(drift_x / 100)
        state_y = int(drift_y / 100)
        return (state_x, state_y)

    def choose_action(self, drift_x, drift_y):
        state = self._get_state_key(drift_x, drift_y)
        
        # Initialize state in Q-table if it's encountered for the first time
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions_count)
            
        # Epsilon-greedy selection strategy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions_count)  # Explore
        else:
            return np.argmax(self.q_table[state])        # Exploit best known action

    def learn(self, drift_x, drift_y, action, reward, next_drift_x, next_drift_y):
        state = self._get_state_key(drift_x, drift_y)
        next_state = self._get_state_key(next_drift_x, next_drift_y)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actions_count)
            
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.actions_count)

         # Bellman Equation update equation for Q-Learning
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.discount * next_max)
        self.q_table[state][action] = new_value

  
    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"--> [AI Agent] Saved {len(self.q_table)} states of experience successfully!")

    def load_q_table(self, filename="q_table.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.q_table = pickle.load(f)
            except (E0FError, Exceptation):
                print(f"--> [AI Agent] loaded {len(self.q.table)} states from past flights! Experienced pilot ready.")
                self.q_table ={}
        else:
            self.q_table = {}
            print("--> [AI Agent] No previous experience file found. Starting fresh as a rookie!!!")
        
            
       

# Initialize our smart agent
ai_agent = QLearningAgent(actions_count=4)
ai_agent.load_q_table()

# 🛸 FLIGHT PATH AND ENVIRONMENT SIMULATION LOOP
# =================================================================
print("--> [9.5] Simulating drone flight path with active RL Correction...")

num_steps = 150
x_true = np.linspace(x_real[0], x_real[-1], num_steps)
y_true = np.linspace(y_real[0], y_real[-1], num_steps)
safe_altitude = np.max(Z_flip) + 300
z_true = np.ones(num_steps) * safe_altitude

true_points = np.column_stack((x_true, y_true, z_true))

# Initialize tracking array for the learning curve
episode_rewards = []

for episode in range(500):
    
    should_render = (episode == 499)  # Render only the final episode to save time
    
    drift_points = []
    ai_corrected_points = []
    current_drift_x = 0
    current_drift_y = 0
    gps_loss_step = 30
    
    # Track total rewards for the current episode
    total_episode_reward = 0

    if should_render:
        plotter.add_lines(true_points, color='green', width=5, label='Planned Route (GPS)')
        drift_line_actor = None
        drone_actor = plotter.add_mesh(pv.Sphere(radius=400, center=true_points[0]), color='yellow', label='Live Drone Position')
        plotter.add_legend()
        plotter.show(interactive_update=True)
        print(f"\n--> [10] SHOWING FINAL LIVE SIMULATION FOR EPISODE {episode + 1}/100...")
    else:
        if (episode + 1) % 10 == 0:
            print(f"--> Training in background... Episode {episode + 1}/500 complete.")

    for step in range(num_steps):
        tx, ty, tz = true_points[step]
        
        # Default baseline reward for safe steps before GPS loss
        reward = 0
        
        if step >= gps_loss_step:
            current_drift_x += np.random.normal(0, 180)
            current_drift_y += np.random.normal(0, 180)
            
            state_before_action_x = current_drift_x
            state_before_action_y = current_drift_y
            
            action = ai_agent.choose_action(current_drift_x, current_drift_y)
            
            correction_step = 120
            if action == 0:    current_drift_x -= correction_step
            elif action == 1:  current_drift_x += correction_step
            elif action == 2:  current_drift_y -= correction_step
            elif action == 3:  current_drift_y += correction_step
            
            current_distance_error = np.sqrt(current_drift_x**2 + current_drift_y**2)
            
            next_drift_x = current_drift_x + np.random.normal(0, 10)
            next_drift_y = current_drift_y + np.random.normal(0, 10)
            next_distance_error = np.sqrt(next_drift_x**2 + next_drift_y**2)
            
            # Continuous Reward Shaping Function
            reward = -next_distance_error
            
            if next_distance_error < current_distance_error:
                reward += 50  # Incentive for reducing the tracking gap
                
            if next_distance_error < 60:
                reward += 200  # Large bonus for absolute stability on track
                
            ai_agent.learn(state_before_action_x, state_before_action_y, action, reward, next_drift_x, next_drift_y)

        # Accumulate reward for the current step
        total_episode_reward += reward

        actual_x = tx + current_drift_x
        actual_y = ty + current_drift_y
        ai_corrected_points.append([actual_x, actual_y, tz])
        
        if should_render:
            if len(ai_corrected_points) > 1:
                if drift_line_actor:
                    plotter.remove_actor(drift_line_actor)
                line_mesh = pv.lines_from_points(np.array(ai_corrected_points))
                drift_line_actor = plotter.add_mesh(line_mesh, color='red', line_width=4)

            new_sphere = pv.Sphere(radius=400, center=[actual_x, actual_y, tz])
            drone_actor.mapper.dataset.copy_from(new_sphere)
            plotter.camera.azimuth += 0.25
            plotter.update()
            time.sleep(0.04)

    # Decay exploration rate and log total episode reward
    ai_agent.epsilon = max(0.05, ai_agent.epsilon * 0.97)
    episode_rewards.append(total_episode_reward)

# --- End of Simulation and Training ---
ai_agent.save_q_table()
print("\n--> [11] AI Training Complete! Preparing plots...")

if 'plotter' in locals():
    plotter.show()

# Plot the learning curve using Matplotlib
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, color='crimson', linewidth=2, label='Total Reward per Episode')
plt.title('Drone AI Learning Curve (Reward Shaping Impact)', fontsize=14, fontweight='bold')
plt.xlabel('Episodes', fontsize=12)
plt.ylabel('Total Accumulated Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
