import gymnasium as gym
import numpy as np
import os
import sys 
import random
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from scipy.optimize import minimize
import matplotlib
# Force non-interactive backend for saving plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. Global Configuration
# ==========================================
# Path to the pre-trained model
MODEL_PATH = "./checkpoints/sac_model_500000_steps.zip" 
SAVE_BASE_DIR = "./polished_solutions"

# Sampling configuration
TOTAL_SAMPLING_STEPS = 100_000 
SAMPLING_FREQ = 1000      # Trigger optimization every 1000 steps
MAX_POLISH_ITER = 5000    # Max iterations for SciPy optimizer
OVERLAP_TOLERANCE = 1e-5  # Tolerance for valid solution (overlap < 1e-5)

# ==========================================
# 2. Utility Functions (SciPy & Lattice Gen)
# ==========================================

def generate_hexagonal_lattice(n_circles):
    """
    Generates an initial hexagonal lattice arrangement for environment reset.
    Ensures circles are neatly packed within the unit square.
    """
    n_side = int(np.ceil(np.sqrt(n_circles))) 
    state = np.zeros(n_circles * 3, dtype=np.float32)
    
    # Initial radius (slightly smaller to avoid jamming)
    r_init = 1.0 / (2 * n_side + 1)
    
    count = 0
    for i in range(n_side + 2):
        for j in range(n_side + 2):
            if count >= n_circles:
                break
            
            x_offset = 0.5 * r_init if j % 2 == 1 else 0
            x = (i + 0.5) * (2 * r_init) + x_offset
            y = (j + 0.5) * (2 * r_init * np.sqrt(3)/2)
            
            if x < 1.0 and y < 1.0:
                idx = count * 3
                state[idx] = x
                state[idx+1] = y
                state[idx+2] = r_init
                count += 1
                
    # Fill remaining spots randomly if any
    while count < n_circles:
        idx = count * 3
        state[idx] = np.random.uniform(0.1, 0.9)
        state[idx+1] = np.random.uniform(0.1, 0.9)
        state[idx+2] = r_init
        count += 1
        
    return state

def overlap_loss(coords, r):
    """ SciPy optimization objective: Minimize total overlap squared. """
    n = len(r)
    x = coords[:n]
    y = coords[n:]
    loss = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            dist = np.sqrt(dist_sq)
            target_dist = r[i] + r[j]
            if dist < target_dist:
                loss += (target_dist - dist)**2 
    return loss

def plot_solution(state, filename, title_suffix=""):
    """ Visualizes and saves the circle packing configuration. """
    n = len(state) // 3
    x = state[0::3]
    y = state[1::3]
    r = state[2::3]
    r_sum = np.sum(r)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    # Draw container boundary
    rect = patches.Rectangle((0,0), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Draw circles with colormap based on radius size
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=np.min(r), vmax=np.max(r))
    
    for i in range(n):
        color = cmap(norm(r[i]))
        circle = patches.Circle((x[i], y[i]), r[i], alpha=0.8, color=color, ec='black', linewidth=1.0)
        ax.add_patch(circle)
        
    plt.title(f"Circle Packing (N={n})\nRadius Sum: {r_sum:.5f} {title_suffix}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

# ==========================================
# 3. Environment Definition
# ==========================================

class PhysicsGuidedEnv(gym.Env):
    def __init__(self, n_circles=26, total_steps=3_000_000):
        super().__init__()
        self.n_circles = n_circles
        self.total_steps = total_steps
        
        # Action: [dx, dy, dr] perturbation
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(n_circles * 3,), dtype=np.float32)
        # State: [x, y, r]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_circles * 3,), dtype=np.float32)
        self.state = None
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.state = generate_hexagonal_lattice(self.n_circles)
        return self.state, {}

    def step(self, action):
        # Apply action as a perturbation (Random Walk exploration)
        self.state = self.state + action 
        
        # Basic clipping
        self.state[2::3] = np.clip(self.state[2::3], 0.01, 0.5) 
        self.state[0::3] = np.clip(self.state[0::3], 0.0, 1.0)
        self.state[1::3] = np.clip(self.state[1::3], 0.0, 1.0)

        r_sum = np.sum(self.state[2::3])
        
        info = {
            'radius_sum': r_sum,
            'penalty': 1.0 # Dummy value
        }
        
        # In pure sampling mode, Reward is 0 and Done is False
        return self.state, 0.0, False, False, info

# ==========================================
# 4. Callback (Quick Check + Deep Polish)
# ==========================================

class SamplingWithPolishCallback(BaseCallback):
    def __init__(self, save_path, sampling_freq, overlap_tol):
        super().__init__(verbose=0)
        self.save_path = save_path
        self.sampling_freq = sampling_freq
        self.overlap_tol = overlap_tol
        os.makedirs(save_path, exist_ok=True)
        self.saved_count = 0
        self.best_global_r = -np.inf
        
        # --- Champion Buffer ---
        self.period_best_state = None
        self.period_best_r = -np.inf
        self.period_best_loss = 999.0 
        
    def _on_step(self) -> bool:
        
        # ============================================================
        # Phase 1: Exploration (Step 1 ~ 999) - Quick Polish Check
        # ============================================================
        try:
            current_state = self.locals["env"].envs[0].unwrapped.state.copy()
            x_curr = current_state[0::3]
            y_curr = current_state[1::3]
            r_curr = current_state[2::3]
            current_r_sum = np.sum(r_curr)
            
            # Only optimize if current R sum is promising
            if current_r_sum > self.period_best_r:
                
                # --- Quick Polish (50 iter) ---
                n = len(r_curr)
                initial_coords = np.concatenate([x_curr, y_curr])
                bounds = []
                for val in r_curr: bounds.append((val, 1.0 - val))
                for val in r_curr: bounds.append((val, 1.0 - val))
                
                res = minimize(
                    fun=overlap_loss, 
                    x0=initial_coords,
                    args=(r_curr,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-5, 'disp': False, 'maxiter': 50} 
                )
                
                quick_loss = res.fun
                
                # If loss is low enough or R is exceptionally high, store it
                if quick_loss < 0.1:
                    self.period_best_r = current_r_sum
                    optimized_state = current_state.copy()
                    optimized_state[0::3] = res.x[:n]
                    optimized_state[1::3] = res.x[n:]
                    self.period_best_state = optimized_state
                    self.period_best_loss = quick_loss

        except Exception:
            pass

        # ============================================================
        # Phase 2: Settlement (Step 1000) - Deep Polish
        # ============================================================
        if self.num_timesteps % self.sampling_freq == 0:
            
            if self.period_best_state is None:
                print(f"Step {self.num_timesteps}: No good candidate found in this period.")
            else:
                target_state = self.period_best_state
                target_r_sum = self.period_best_r
                print(f"Step {self.num_timesteps}: [TRIGGER] Best Candidate R: {target_r_sum:.4f} (Pre-Loss: {self.period_best_loss:.4f}) -> Deep Polish...")
                sys.stdout.flush()

                x_start = target_state[0::3]
                y_start = target_state[1::3]
                r_original = target_state[2::3]
                
                # --- Deep Polish Config ---
                max_retries = 40       
                shrink_step = 0.995    # Shrink 0.5% per retry
                success = False
                scale_factor = 1.0
                best_loss_in_loop = 999.0
                
                for attempt in range(max_retries):
                    r_current = r_original * scale_factor
                    current_max_iter = MAX_POLISH_ITER if attempt == 0 else 1000
                    
                    try:
                        n = len(r_current)
                        initial_coords = np.concatenate([x_start, y_start])
                        bounds = []
                        for val in r_current: bounds.append((val, 1.0 - val))
                        for val in r_current: bounds.append((val, 1.0 - val))
                        
                        # --- Jitter Mechanism ---
                        if attempt > 0:
                            jitter_scale = 0.002 * (attempt / 5.0) 
                            noise = np.random.uniform(-jitter_scale, jitter_scale, size=initial_coords.shape)
                            x0_input = np.clip(initial_coords + noise, 0, 1)
                            # Re-clip to bounds
                            lb = np.array([b[0] for b in bounds])
                            ub = np.array([b[1] for b in bounds])
                            x0_input = np.clip(x0_input, lb, ub)
                        else:
                            x0_input = initial_coords

                        res = minimize(
                            fun=overlap_loss, 
                            x0=x0_input,
                            args=(r_current,),
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'ftol': 1e-9, 'disp': False, 'maxiter': current_max_iter}
                        )
                        
                        x_final, y_final, loss = res.x[:n], res.x[n:], res.fun
                        
                        if loss < best_loss_in_loop: best_loss_in_loop = loss
                        
                        if loss < self.overlap_tol:
                            # SUCCESS
                            success = True
                            final_r_sum = np.sum(r_current)
                            self.saved_count += 1
                            
                            is_new_best = False
                            if final_r_sum > self.best_global_r:
                                self.best_global_r = final_r_sum
                                is_new_best = True
                            
                            # Save state
                            final_state = np.concatenate([x_final, y_final, r_current])
                            filename = f"R{final_r_sum:.5f}_step{self.num_timesteps}_#{self.saved_count}.npy"
                            full_path = os.path.join(self.save_path, filename)
                            np.save(full_path, final_state)
                            
                            # Plot result
                            plot_filename = full_path.replace('.npy', '.png')
                            plot_solution(final_state, plot_filename, title_suffix=f"(Step {self.num_timesteps})")
                            
                            color = "\033[92m" if is_new_best else ""
                            reset = "\033[0m"
                            print(f"{color}Step {self.num_timesteps}: ✅ SUCCESS (Scale={scale_factor:.3f}) | Score: {final_r_sum:.5f}{reset}")
                            sys.stdout.flush()
                            break 
                        
                        else:
                            # FAIL, Shrink and Retry
                            scale_factor *= shrink_step 
                            x_start, y_start = x_final, y_final
                            
                    except Exception:
                        pass

                if not success:
                    print(f"Step {self.num_timesteps}: ❌ Failed (Best Loss: {best_loss_in_loop:.4f})")
                    sys.stdout.flush()
            
            # Reset Buffer
            self.period_best_state = None
            self.period_best_r = -np.inf
            self.period_best_loss = 999.0
                
        return True

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed initialized to {seed}")

    # Prepare directories
    os.makedirs(SAVE_BASE_DIR, exist_ok=True)
    
    # Check Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    print(f"Loading model from {MODEL_PATH}...")
    
    # Init Environment
    env = Monitor(PhysicsGuidedEnv(n_circles=26, total_steps=TOTAL_SAMPLING_STEPS))
    
    # Load SAC Model
    model = SAC.load(MODEL_PATH, env=env, device="cuda")
    
    # Switch to Evaluation/Exploration Mode
    model.set_env(env)
    model.policy.set_training_mode(False)
    
    # Init Callback
    callback = SamplingWithPolishCallback(
        save_path=SAVE_BASE_DIR,
        sampling_freq=SAMPLING_FREQ,
        overlap_tol=OVERLAP_TOLERANCE
    )
    
    print(f"Start sampling for {TOTAL_SAMPLING_STEPS} steps...")
    
    # Start Learning Loop
    model.learn(
        total_timesteps=TOTAL_SAMPLING_STEPS,
        callback=callback,
        reset_num_timesteps=False
    )
    
    print("Sampling Completed.")