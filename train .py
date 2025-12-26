import gymnasium as gym
import numpy as np
import matplotlib
# Force non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import torch
import sys 

# --- 1. Geometric Initialization ---

def generate_hexagonal_lattice(n=26, noise_level=0.02):
    """ Generates a perturbed triangular lattice initialization. """
    points = []
    rows = 6
    cols = 6
    for r in range(rows):
        for c in range(cols):
            x_offset = 0.5 if r % 2 != 0 else 0.0
            x = (c + x_offset) * 0.18 + 0.05
            y = r * 0.15 + 0.05
            if 0 < x < 1 and 0 < y < 1:
                points.append([x, y])
    points = np.array(points)
    if len(points) < n:
        extra = np.random.rand(n - len(points), 2)
        points = np.vstack([points, extra])
    center = np.array([0.5, 0.5])
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points += np.random.uniform(-noise_level, noise_level, size=points.shape)
    points = np.clip(points, 0.01, 0.99)
    
    # Initial radius set to 0.03
    radii = np.ones((n, 1)) * 0.03 
    state = np.hstack([points, radii]).flatten()
    return state.astype(np.float32)

# --- 2. Physics Guided Environment ---

class PhysicsGuidedEnv(gym.Env):
    def __init__(self, n_circles=26, total_steps=500_000):
        super(PhysicsGuidedEnv, self).__init__()
        self.n_circles = n_circles
        self.total_steps = total_steps
        self.global_step_counter = 0 
        
        # Target penalty weight
        self.TARGET_LAMBDA = 100.0 
        
        # Hard constraint for minimum radius
        self.MIN_RADIUS = 0.03
        
        # Physics repulsion gains
        self.PHYSICS_REPULSION_GAIN = 1.5
        self.PHYSICS_WALL_GAIN = 1.5
        
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(n_circles * 3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_circles * 3,), dtype=np.float32)
        
        self.state = None
        self.best_sum = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = generate_hexagonal_lattice(self.n_circles)
        return self.state, {}

    def step(self, action):
        self.global_step_counter += 1
        
        # Lambda schedule: Peak at 50%
        # 0% - 10%: Lambda = 0 (Free expansion)
        # 10% - 50%: Linear increase to TARGET_LAMBDA
        # 50%+: Constant TARGET_LAMBDA
        progress = self.global_step_counter / self.total_steps
        if progress < 0.1:
            current_lambda = 0.0
        elif progress < 0.5:
            ratio = (progress - 0.1) / 0.4 
            current_lambda = ratio * self.TARGET_LAMBDA
        else:
            current_lambda = self.TARGET_LAMBDA
            
        # --- Physics Engine Calculation ---
        x = self.state[0::3]
        y = self.state[1::3]
        r = self.state[2::3]
        
        dx_agent = action[0::3]
        dy_agent = action[1::3]
        dr_agent = action[2::3]

        # Inter-particle repulsion
        delta_x = x[:, None] - x[None, :]
        delta_y = y[:, None] - y[None, :]
        dists = np.sqrt(delta_x**2 + delta_y**2)
        np.fill_diagonal(dists, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        overlaps = np.maximum(0, r_sum - dists)
        
        force_x = np.sum((delta_x / (dists + 1e-6)) * overlaps, axis=1) * self.PHYSICS_REPULSION_GAIN
        force_y = np.sum((delta_y / (dists + 1e-6)) * overlaps, axis=1) * self.PHYSICS_REPULSION_GAIN
        
        # Wall repulsion
        wall_force_x = np.zeros_like(x)
        wall_force_y = np.zeros_like(y)
        wall_force_x += np.maximum(0, r - x) * self.PHYSICS_WALL_GAIN
        wall_force_x -= np.maximum(0, x + r - 1.0) * self.PHYSICS_WALL_GAIN
        wall_force_y += np.maximum(0, r - y) * self.PHYSICS_WALL_GAIN
        wall_force_y -= np.maximum(0, y + r - 1.0) * self.PHYSICS_WALL_GAIN
        
        # Update state
        new_x = x + dx_agent + force_x + wall_force_x
        new_y = y + dy_agent + force_y + wall_force_y
        new_r = np.clip(r + dr_agent, self.MIN_RADIUS, 0.4)
        
        self.state[0::3] = np.clip(new_x, 0.0, 1.0)
        self.state[1::3] = np.clip(new_y, 0.0, 1.0)
        self.state[2::3] = new_r
        
        # --- Reward Calculation ---
        x, y, r = self.state[0::3], self.state[1::3], self.state[2::3]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, None] + r[None, :]
        np.fill_diagonal(dists, np.inf)
        
        final_overlaps = np.maximum(0, r_sum - dists)
        penalty_overlap = np.sum(final_overlaps**2) * 0.5 
        
        p_b_x_high = np.maximum(0, x + r - 1.0)
        p_b_x_low  = np.maximum(0, r - x)
        p_b_y_high = np.maximum(0, y + r - 1.0)
        p_b_y_low  = np.maximum(0, r - y)
        penalty_boundary = np.sum(p_b_x_high**2 + p_b_x_low**2 + p_b_y_high**2 + p_b_y_low**2)

        r_total = np.sum(r)
        
        # Reward formula
        reward = (r_total * 100.0) - (current_lambda * (penalty_overlap + penalty_boundary))
        
        is_structurally_feasible = (penalty_overlap < 0.1 and penalty_boundary < 0.1)
        
        if is_structurally_feasible and r_total > self.best_sum:
            self.best_sum = r_total

        terminated = False
        truncated = False
        info = {
            "radius_sum": r_total,
            "penalty": penalty_overlap + penalty_boundary,
            "lambda": current_lambda,
            "is_feasible": is_structurally_feasible
        }
        return self.state, reward, terminated, truncated, info

# --- 3. Callbacks and Utils ---

class LoggingCallback(BaseCallback):
    def __init__(self, save_path, save_freq):
        super().__init__(verbose=0)
        self.save_path = save_path
        self.save_freq = save_freq
        self.best_radius_sum = -np.inf
        self.best_state_to_save = None

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        info = infos[0]
        current_r = info['radius_sum']
        
        if info['is_feasible'] and current_r > self.best_radius_sum:
            self.best_radius_sum = current_r
            self.best_state_to_save = self.locals["env"].envs[0].unwrapped.state.copy()

        # Save Best Model Result
        if self.num_timesteps % self.save_freq == 0 and self.best_state_to_save is not None:
            np.save(f"{self.save_path}/best_result.npy", self.best_state_to_save)
            
        # Logging
        if self.num_timesteps % 1000 == 0:
             penalty = info['penalty']
             lam = info['lambda']
             print(f"Step: {self.num_timesteps} | Lambda: {lam:.1f} | Curr R: {current_r:.4f} | Pen: {penalty:.4f} | Best: {self.best_radius_sum:.4f}")
             sys.stdout.flush()

        return True

def plot_packing(npy_file):
    if not os.path.exists(npy_file):
        return
    state = np.load(npy_file)
    n = len(state) // 3
    x = state[0::3]
    y = state[1::3]
    r = state[2::3]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    rect = patches.Rectangle((0,0), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    colors = plt.cm.viridis(r / np.max(r))
    for i in range(n):
        circle = patches.Circle((x[i], y[i]), r[i], alpha=0.8, color=colors[i], ec='black')
        ax.add_patch(circle)
        
    total_r = np.sum(r)
    save_path = f"final_result_R{total_r:.4f}.png"
    plt.title(f"Optimized Result\nR={total_r:.5f}")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # Settings
    TOTAL_STEPS = 500_000
    SAVE_FREQ = 10000
    SEED = 42

    # Directories
    save_dir = "./results"
    ckpt_dir = "./checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Environment Setup
    env = Monitor(PhysicsGuidedEnv(n_circles=26, total_steps=TOTAL_STEPS))
    env.reset(seed=SEED)
    
    print(f"Training started on device: {torch.cuda.current_device()} with seed {SEED}")
    sys.stdout.flush()

    # SAC Configuration
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=512,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=4,
        verbose=0,
        seed=SEED,
        device="cuda"
    )

    # Callbacks: Logging + Checkpointing
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=ckpt_dir,
        name_prefix="sac_model"
    )
    
    log_callback = LoggingCallback(save_path=save_dir, save_freq=SAVE_FREQ)
    
    callback = CallbackList([log_callback, checkpoint_callback])

    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    
    print("Training Finished.")
    sys.stdout.flush()
    
    best_file = f"{save_dir}/best_result.npy"
    if os.path.exists(best_file):
        plot_packing(best_file)