import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TinyPolicy

# ---------------------------------------------------------
# STEP 3: RUN POLICY IN SIMULATION (CLOSED-LOOP ROLLOUT)
# ---------------------------------------------------------

class SimulatedFrankaWorkspace:
    """
    Simulated Franka Panda Tabletop Manipulation Environment.
    Models Cartesian End-Effector dynamics, workspace bounds, and object interactions.
    """
    def __init__(self, cube_pos=(0.733, 0.354, 0.05), bowl_pos=(0.243, 0.352, 0.05)):
        self.cube_pos = np.array(cube_pos, dtype=np.float32)
        self.bowl_pos = np.array(bowl_pos, dtype=np.float32)
        self.reset()

    def reset(self):
        # Initial robot end-effector starting pose matching demonstration distribution
        self.eef_pos = np.array([0.82, 1.01, 0.0], dtype=np.float32)
        self.gripper_state = 0.0 # 0.0 = Open, 1.0 = Closed
        self.cube_held = False
        self.step_count = 0
        return self.get_observation()

    def get_observation(self):
        return {
            "hand_pos": self.eef_pos.copy(),
            "cube_pos": self.cube_pos[:2].copy(),
            "bowl_pos": self.bowl_pos[:2].copy(),
            "gripper": self.gripper_state
        }

    def step(self, action_pred):
        dx, dy, dz, grip_action = action_pred
        self.step_count += 1

        # Apply displacement deltas
        self.eef_pos[0] += dx
        self.eef_pos[1] += dy
        self.eef_pos[2] += dz

        # Threshold gripper trigger
        self.gripper_state = 1.0 if grip_action > 0.5 else 0.0

        # Physical interaction heuristics:
        # Grasp check: if gripper closes within 5cm of cube, grasp cube
        dist_to_cube = np.linalg.norm(self.eef_pos - self.cube_pos)
        if self.gripper_state > 0.5 and dist_to_cube < 0.08:
            self.cube_held = True

        # Carry cube with end-effector if held
        if self.cube_held:
            self.cube_pos = self.eef_pos.copy() - np.array([0, 0, 0.03])

        # Release check: if gripper opens over bowl
        dist_to_bowl = np.linalg.norm(self.cube_pos[:2] - self.bowl_pos[:2])
        if self.gripper_state < 0.5 and self.cube_held:
            self.cube_held = False
            # If released near bowl, cube settles into bowl
            if dist_to_bowl < 0.12:
                self.cube_pos = self.bowl_pos.copy()

        # Check task completion
        success = (not self.cube_held) and (np.linalg.norm(self.cube_pos[:2] - self.bowl_pos[:2]) < 0.08)
        done = success or (self.step_count >= 120)

        return self.get_observation(), success, done

def run_simulation(checkpoint="policy.pt", render_plot=True):
    print("Loading Trained Policy...")
    model = TinyPolicy(input_dim=7, hidden_dim=128, output_dim=4)
    if not os.path.exists(checkpoint):
        print(f"Error: {checkpoint} not found. Run 2_train_mlp.py first.")
        return

    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    print("Policy successfully loaded.")

    print("\nInitializing Robot Manipulation Simulation...")
    env = SimulatedFrankaWorkspace()
    obs = env.reset()

    traj_x, traj_y, traj_z, grippers = [], [], [], []

    print(f"Starting Closed-Loop Rollout...")
    success = False
    for step in range(120):
        hx, hy, hz = obs["hand_pos"]
        cx, cy = obs["cube_pos"]
        bx, by = obs["bowl_pos"]

        traj_x.append(hx)
        traj_y.append(hy)
        traj_z.append(hz)
        grippers.append(obs["gripper"])

        # Construct input state vector: [hand_x, hand_y, hand_z, cube_x, cube_y, bowl_x, bowl_y]
        state_vec = np.array([hx, hy, hz, cx, cy, bx, by], dtype=np.float32)
        state_tensor = torch.tensor(state_vec).unsqueeze(0)

        # Policy forward pass
        with torch.no_grad():
            action_raw = model(state_tensor).squeeze().numpy()

        dx, dy, dz, grip_logit = action_raw
        grip_prob = 1.0 / (1.0 + np.exp(-grip_logit)) # Sigmoid

        action = (dx, dy, dz, grip_prob)
        obs, success, done = env.step(action)

        if step % 20 == 0 or done:
            print(f"Step {step:03d} | EEF Pos: [{hx:.3f}, {hy:.3f}, {hz:.3f}] | Grip: {obs['gripper']:.0f} | Cube Held: {env.cube_held}")

        if done:
            break

    print("\n================ EVALUATION OUTCOME ================")
    print(f"Task: Pick up Rubik's Cube and place into Bowl")
    print(f"Total Steps Taken: {len(traj_x)}")
    print(f"Final End-Effector Position: [{traj_x[-1]:.3f}, {traj_y[-1]:.3f}, {traj_z[-1]:.3f}]")
    print(f"Final Cube Position: [{env.cube_pos[0]:.3f}, {env.cube_pos[1]:.3f}, {env.cube_pos[2]:.3f}]")
    print(f"Target Bowl Position: [{env.bowl_pos[0]:.3f}, {env.bowl_pos[1]:.3f}, {env.bowl_pos[2]:.3f}]")
    print(f"Status: {'SUCCESS - TASK COMPLETED!' if success else 'REACHED MAX TIMESTEPS'}")
    print("====================================================")

    if render_plot:
        plt.figure(figsize=(10, 6))
        # Plot 2D top-down view
        plt.plot(traj_x, traj_y, 'b-', label='Robot EEF Trajectory', linewidth=2)
        plt.scatter([traj_x[0]], [traj_y[0]], color='green', s=100, label='Start')
        plt.scatter([traj_x[-1]], [traj_y[-1]], color='purple', s=100, label='End')
        plt.scatter([0.733], [0.354], color='red', marker='s', s=150, label='Pick Zone (Cube ID 0)')
        plt.scatter([0.243], [0.352], color='orange', marker='o', s=180, label='Place Zone (Bowl ID 1)')

        # Color segments where gripper was closed
        for i in range(len(traj_x) - 1):
            if grippers[i] > 0.5:
                plt.plot(traj_x[i:i+2], traj_y[i:i+2], 'r-', linewidth=3.5)

        plt.title("Closed-Loop Robot Manipulation Rollout (Egocentric Behavioral Cloning)", fontsize=13)
        plt.xlabel("Workspace X (Normalized Table Width)")
        plt.ylabel("Workspace Y (Normalized Table Depth)")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("rollout_trajectory.png", dpi=150)
        print("Trajectory plot saved to 'rollout_trajectory.png'.")

if __name__ == "__main__":
    run_simulation()
