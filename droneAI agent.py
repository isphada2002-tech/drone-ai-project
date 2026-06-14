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
# 🧠 DEEP Q-NETWORK (DQN) AGENT IMPLEMENTATION FROM SCRATCH (NUMPY ONLY)
# =================================================================
class DeepQLearningAgent:
    def __init__(self, actions_count=4, input_dim=2, hidden_dim=16):
        self.actions_count = actions_count
        self.input_dim = input_dim      # Input features: [drift_x, drift_y]
        self.hidden_dim = hidden_dim    # Number of hidden layer neurons
        
        self.lr = 0.01        # Learning Rate for weight updates
        self.discount = 0.95  # Gamma discount factor
        self.epsilon = 0.3    # Exploration rate for epsilon-greedy policy

        # 🎲 Weight and Bias Initialization (He Initialization optimized for ReLU)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        
        self.W2 = np.random.randn(self.hidden_dim, self.actions_count) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, self.actions_count))

    def _prepare_input(self, drift_x, drift_y):
        # Feature Scaling: Normalize inputs to prevent gradient explosion
        # and keep matrix operations numerically stable.
        return np.array([[drift_x / 1000.0, drift_y / 1000.0]], dtype=float)

    def forward_pass(self, X):
        # Forward pass calculation through the layers
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)  # Element-wise ReLU activation function
        Q_values = np.dot(A1, self.W2) + self.b2
        return Q_values, Z1, A1

    def choose_action(self, drift_x, drift_y):
        X = self._prepare_input(drift_x, drift_y)
        Q_values, _, _ = self.forward_pass(X)

        # Epsilon-Greedy action selection strategy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions_count)  # Explore random action
        else:
            return np.argmax(Q_values[0])  # Exploit the action with highest predicted Q-value

    def learn(self, drift_x, drift_y, action, reward, next_drift_x, next_drift_y):
        # 1. Prepare continuous spatial matrices for current and next state
        X = self._prepare_input(drift_x, drift_y)
        X_next = self._prepare_input(next_drift_x, next_drift_y)

        # 2. Get current predictions and next state max Q-values
        Q_values, Z1, A1 = self.forward_pass(X)
        Q_next, _, _ = self.forward_pass(X_next)

        # 3. Calculate target using the Bellman Optimality Equation
        target_Q = Q_values.copy()
        max_next_q = np.max(Q_next[0])
        target_Q[0][action] = reward + self.discount * max_next_q

        # 4. Calculate Loss Gradient with respect to Output Layer
        dL_dQ = Q_values - target_Q  # Shape: (1, actions_count)
        dL_dQ = np.clip(dL_dQ, -1.0, 1.0)

        # 5. Backpropagation: Compute gradients using mathematical Chain Rule
        dW2 = np.dot(A1.T, dL_dQ)
        db2 = np.sum(dL_dQ, axis=0, keepdims=True)

        dA1 = np.dot(dL_dQ, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU activation

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 6. Update Neural Network parameters via Stochastic Gradient Descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def save_weights(self, filename="dqn_weights.pkl"):
        # Save deep neural network weights instead of a table dictionary
        weights = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        with open(filename, "wb") as f:
            pickle.dump(weights, f)
        print("--> [AI Agent] Saved Deep Neural Network weights successfully!")

    def load_weights(self, filename="dqn_weights.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    weights = pickle.load(f)
                self.W1 = weights["W1"]
                self.b1 = weights["b1"]
                self.W2 = weights["W2"]
                self.b2 = weights["b2"]
                print("--> [AI Agent] Deep Neural Network weights loaded! Experienced brain active.")
            except Exception:
                print("--> [AI Agent] Error loading weights. Starting with a fresh network architecture!")
        else:
            print("--> [AI Agent] No previous weights file found. Initializing a fresh neural brain!")
        
            
       

# Initialize our smart agent
ai_agent = DeepQLearningAgent(actions_count=4, input_dim=2, hidden_dim=16)
ai_agent.load_weights()
ai_agent.save_weights()

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
        print(f"\n--> [10] SHOWING FINAL LIVE SIMULATION FOR EPISODE {episode + 1}/500...")
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
            reward = -next_distance_error / 100.0
            
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
# 🧠 DEEP Q-NETWORK (DQN) AGENT IMPLEMENTATION FROM SCRATCH (NUMPY ONLY)
# =================================================================
class DeepQLearningAgent:
    def __init__(self, actions_count=4, input_dim=2, hidden_dim=16):
        self.actions_count = actions_count
        self.input_dim = input_dim      # Input features: [drift_x, drift_y]
        self.hidden_dim = hidden_dim    # Number of hidden layer neurons
        
        self.lr = 0.01        # Learning Rate for weight updates
        self.discount = 0.95  # Gamma discount factor
        self.epsilon = 0.3    # Exploration rate for epsilon-greedy policy

        # 🎲 Weight and Bias Initialization (He Initialization optimized for ReLU)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        
        self.W2 = np.random.randn(self.hidden_dim, self.actions_count) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, self.actions_count))

    def _prepare_input(self, drift_x, drift_y):
        # Feature Scaling: Normalize inputs to prevent gradient explosion
        # and keep matrix operations numerically stable.
        return np.array([[drift_x / 1000.0, drift_y / 1000.0]], dtype=float)

    def forward_pass(self, X):
        # Forward pass calculation through the layers
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)  # Element-wise ReLU activation function
        Q_values = np.dot(A1, self.W2) + self.b2
        return Q_values, Z1, A1

    def choose_action(self, drift_x, drift_y):
        X = self._prepare_input(drift_x, drift_y)
        Q_values, _, _ = self.forward_pass(X)

        # Epsilon-Greedy action selection strategy
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions_count)  # Explore random action
        else:
            return np.argmax(Q_values[0])  # Exploit the action with highest predicted Q-value

    def learn(self, drift_x, drift_y, action, reward, next_drift_x, next_drift_y):
        # 1. Prepare continuous spatial matrices for current and next state
        X = self._prepare_input(drift_x, drift_y)
        X_next = self._prepare_input(next_drift_x, next_drift_y)

        # 2. Get current predictions and next state max Q-values
        Q_values, Z1, A1 = self.forward_pass(X)
        Q_next, _, _ = self.forward_pass(X_next)

        # 3. Calculate target using the Bellman Optimality Equation
        target_Q = Q_values.copy()
        max_next_q = np.max(Q_next[0])
        target_Q[0][action] = reward + self.discount * max_next_q

        # 4. Calculate Loss Gradient with respect to Output Layer
        dL_dQ = Q_values - target_Q  # Shape: (1, actions_count)
        dL_dQ = np.clip(dL_dQ, -1.0, 1.0)

        # 5. Backpropagation: Compute gradients using mathematical Chain Rule
        dW2 = np.dot(A1.T, dL_dQ)
        db2 = np.sum(dL_dQ, axis=0, keepdims=True)

        dA1 = np.dot(dL_dQ, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)  # Derivative of ReLU activation

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 6. Update Neural Network parameters via Stochastic Gradient Descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def save_weights(self, filename="dqn_weights.pkl"):
        # Save deep neural network weights instead of a table dictionary
        weights = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        with open(filename, "wb") as f:
            pickle.dump(weights, f)
        print("--> [AI Agent] Saved Deep Neural Network weights successfully!")

    def load_weights(self, filename="dqn_weights.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    weights = pickle.load(f)
                self.W1 = weights["W1"]
                self.b1 = weights["b1"]
                self.W2 = weights["W2"]
                self.b2 = weights["b2"]
                print("--> [AI Agent] Deep Neural Network weights loaded! Experienced brain active.")
            except Exception:
                print("--> [AI Agent] Error loading weights. Starting with a fresh network architecture!")
        else:
            print("--> [AI Agent] No previous weights file found. Initializing a fresh neural brain!")
        
            
       

# Initialize our smart agent
ai_agent = DeepQLearningAgent(actions_count=4, input_dim=2, hidden_dim=16)
ai_agent.load_weights()
ai_agent.save_weights()

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
        print(f"\n--> [10] SHOWING FINAL LIVE SIMULATION FOR EPISODE {episode + 1}/500...")
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
            reward = -next_distance_error / 100.0
            
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

