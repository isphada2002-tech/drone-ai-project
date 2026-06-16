import numpy as np
import pickle
import os

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
# ========================================================================
# 🧠 UPGRADED DOUBLE DQN AGENT WITH 3D SPATIAL INPUTS (NUMPY ONLY)
# ========================================================================
class DoubleDQNAgent:
    def __init__(self, actions_count=6, input_dim=3, hidden_dim=16, target_update_freq=100):
        self.actions_count = actions_count
        self.input_dim = input_dim      # 3 Inputs: [drift_x, drift_y, ground_clearance]
        self.hidden_dim = hidden_dim    # Hidden layer dimension
        
        self.lr = 0.001        # Learning Rate
        self.discount = 0.95  # Gamma discount factor
        self.epsilon = 0.3    # Exploration rate

        self.steps_counter = 0
        self.target_update_freq = target_update_freq

        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 32

        # 🎲 Online Network Weights Initialization (He Initialization)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, self.actions_count) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, self.actions_count))

        # 🎯 Target Network Weights Initialization
        self.update_target_network()

    def update_target_network(self):
        """Synchronizes target network weights with online network weights."""
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()

    def _prepare_input(self, drift_x, drift_y, ground_clearance):
        """Continuous Feature Scaling to prevent numerical gradient explosion."""
        # Fixed: Flattened to 1D vector to allow flexible tensor batching without nesting dimensions
        return np.array([drift_x / 1000.0, drift_y / 1000.0, ground_clearance / 100.0], dtype=float)

    def forward_pass(self, X, target=False):
        """Computes forward pass through network layers."""
        # Guard rail: Securely shape input to 2D matrix whether it's a single state or a full batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
            
        w1, b1, w2, b2 = (self.W1_target, self.b1_target, self.W2_target, self.b2_target) if target else (self.W1, self.b1, self.W2, self.b2)
        Z1 = np.dot(X, w1) + b1
        A1 = np.maximum(0, Z1)  # ReLU Activation Function
        Q_values = np.dot(A1, w2) + b2
        return Q_values, Z1, A1

    def choose_action(self, drift_x, drift_y, ground_clearance):
        """Epsilon-Greedy action selection strategy (6 Degrees of Freedom)."""
        X = self._prepare_input(drift_x, drift_y, ground_clearance)
        Q_values, _, _ = self.forward_pass(X, target=False)
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions_count)
        else:
            return np.argmax(Q_values[0])

    def store_transition(self, drift_x, drift_y, ground_clearance, action, reward, next_drift_x, next_drift_y, next_ground_clearance):
        state = self._prepare_input(drift_x, drift_y, ground_clearance)
        next_state = self._prepare_input(next_drift_x, next_drift_y, next_ground_clearance)
        self.memory.push(state, action, reward, next_state)

    def replay_and_learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps_counter += 1
        mini_batch = self.memory.sample(self.batch_size)

        # Enforce exact matrix conversion without hidden or nested single-dimensions
        states = np.array([t[0] for t in mini_batch]).reshape(self.batch_size, -1)
        actions = [t[1] for t in mini_batch]
        rewards = np.array([t[2] for t in mini_batch])
        next_states = np.array([t[3] for t in mini_batch]).reshape(self.batch_size, -1)

        Q_values, Z1, A1 = self.forward_pass(states, target=False)
        Q_next_online, _, _ = self.forward_pass(next_states, target=False)
        Q_next_target, _, _ = self.forward_pass(next_states, target=True)

        target_Q = Q_values.copy()

        for i in range(self.batch_size):
            best_next_action = np.argmax(Q_next_online[i])
            double_q_target_val = Q_next_target[i][best_next_action]
            target_Q[i][actions[i]] = rewards[i] + self.discount * double_q_target_val

        dL_dQ = (Q_values - target_Q) / self.batch_size
        dL_dQ = np.clip(dL_dQ, -1.0, 1.0)

        dW2 = np.dot(A1.T, dL_dQ)
        db2 = np.sum(dL_dQ, axis=0, keepdims=True)
        dA1 = np.dot(dL_dQ, self.W2.T)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(states.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        if self.steps_counter % self.target_update_freq == 0:
            self.update_target_network()

    def save_weights(self, filename="ddqn_weights.pkl"):
        """Saves current network parameters to binary file."""
        weights = {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}
        with open(filename, "wb") as f:
            pickle.dump(weights, f)
        print("--> [AI Agent] Saved 3D Double DQN weights successfully!")

    def load_weights(self, filename="ddqn_weights.pkl"):
        """Loads experienced network parameters from file."""
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    weights = pickle.load(f)
                self.W1 = weights["W1"]
                self.b1 = weights["b1"]
                self.W2 = weights["W2"]
                self.b2 = weights["b2"]
                self.update_target_network()
                print("--> [AI Agent] 3D DDQN brain loaded and active.")
            except Exception:
                print("--> [AI Agent] Error loading weights. Initializing fresh network!")
