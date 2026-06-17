# 🛸 Drone Navigation & Drift Correction using Deep Reinforcement Learning (DDQN)


An autonomous flight simulation powered by Reinforcement Learning (RL) designed to correct drone flight trajectory drift over complex topographic environments without relying on GPS stabilization.

## 🧠 Advanced 3D Double DQN Architecture (Pure NumPy)

Unlike standard implementations that rely on high-level deep learning frameworks, this agent features a custom-built **Double Deep Q-Network (DDQN)** engineered entirely from scratch using **pure NumPy matrix operations**. This design guarantees zero framework overhead, making it highly optimized for resource-constrained embedded systems and low-latency flight controllers.

### 🚀 Key Technical Architectures:
* **Mathematical Backpropagation from Scratch:** Implemented manual forward and backward propagation passes. Gradient calculations (dW1, db1, dW2, db2) are derived directly via the chain rule, utilizing a customized gradient clipping mechanism (+/- 1.0) on the loss gradient (dL/dQ) to securely mitigate the risk of exploding gradients.
* **Overestimation Bias Mitigation:** Leverages a decoupled dual-network setup (Online Network and Target Network). The Online Network selects the optimal 3D action vector, while the Target Network evaluates its corresponding Q-value, effectively eliminating the standard DQN overestimation bias.
* **6-DoF Control & Spatial Feature Scaling:** Navigates the agent dynamically across the 3D grid (X, Y, Z axes for Forward/Backward, Left/Right, Up/Down). Input features `[drift_x, drift_y, ground_clearance]` undergo continuous min-max and feature scaling directly within the state preparation pipeline to ensure numerical stability during neural weight updates.



---

## 📸 Simulation Preview
![Simulation](assets/simulation.png)

## 📊 Drone AI Learning Curve
![Learning Curve](assets/learning_curve.png)

---

## 🌍 Key Features
* **Real-World Topography:** Utilizes actual Digital Elevation Model (DEM) data via **NASA SRTM GL1** for the mountainous region of **Tangier-Tétouan, Morocco** (`output_SRTMGL1.tif`).
* **Physics-Informed RL:** Integrates aerodynamics and gravity constraints to simulate real-world environmental drift.
* **Data Optimization:** Safely downsampled original DEM dimensions (1075 x 2367) into an optimized **54 x 119 grid** for fluid 3D rendering using **PyVista**.
* **Autonomous Adaptation:** Trains a Double DQN agent to actively predict and counter dynamic drift vectors, saving optimized neural parameters into an exportable weights file (`ddqn_weights.pkl`).

## 🛠️ Technical Stack
* **Language:** Python 3.9+
* **Core Libraries:** NumPy, Pandas, Matplotlib
* **Geospatial & Vision:** Pillow / Geospatial Array Parsing / VTK Structured Grid

## 🚀 How to Run Locally
1. Clone the repository:
```bash
   git clone [https://github.com/isphada2002-tech/drone-ai-project.git](https://github.com/isphada2002-tech/drone-ai-project.git)
   cd drone-ai-project
2. Run the main training agent:
´´´bash
    python "droneAI agent.py"



