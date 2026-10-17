"""
3_run_libero.py
===============
Closed-loop rollout in a simulated Franka Panda workspace.

Supports two policy modes:
  --policy mlp      : TinyPolicy MLP (trained in step 2, CPU-only, fast)
  --policy smolvla  : Fine-tuned SmolVLA checkpoint (step 6)
  --policy both     : Run both sequentially and compare results

The simulation renders synthetic top-down workspace frames using OpenCV so
SmolVLA receives visual input compatible with what it was trained on.

Usage
-----
  python 3_run_libero.py                         # MLP (default)
  python 3_run_libero.py --policy smolvla
  python 3_run_libero.py --policy both
  python 3_run_libero.py --policy mlp --trials 5
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from model import TinyPolicy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMOLVLA_CHECKPOINT = (
    "smolvla_checkpoint-20261010T074618Z-1-001"
    "/smolvla_checkpoint/checkpoints/010000/pretrained_model"
)
TASK_DESCRIPTION = "Pick up the Rubik's cube and place it inside the bowl"
IMG_SIZE         = 224
MAX_STEPS        = 120

# ---------------------------------------------------------------------------
# Simulated Franka Workspace
# ---------------------------------------------------------------------------

class SimulatedFrankaWorkspace:
    """
    Simulated Franka Panda Tabletop Manipulation Environment.
    Models Cartesian End-Effector dynamics, workspace bounds, and object interactions.
    Coordinates are normalised to [0, 1] matching the egocentric video space.
    """
    def __init__(self,
                 cube_pos=(0.733, 0.354, 0.05),
                 bowl_pos=(0.243, 0.352, 0.05)):
        self.cube_pos_init = np.array(cube_pos, dtype=np.float32)
        self.bowl_pos_init = np.array(bowl_pos, dtype=np.float32)
        self.reset()

    def reset(self):
        self.eef_pos      = np.array([0.82, 1.01, 0.0], dtype=np.float32)
        self.gripper_state = 0.0
        self.cube_pos     = self.cube_pos_init.copy()
        self.bowl_pos     = self.bowl_pos_init.copy()
        self.cube_held    = False
        self.step_count   = 0
        return self.get_observation()

    def get_observation(self):
        return {
            "hand_pos":  self.eef_pos.copy(),
            "cube_pos":  self.cube_pos[:2].copy(),
            "bowl_pos":  self.bowl_pos[:2].copy(),
            "gripper":   self.gripper_state,
        }

    def get_state_vector(self):
        """Return the 7-dim state vector used by both policies."""
        hx, hy, hz = self.eef_pos
        cx, cy     = self.cube_pos[:2]
        bx, by     = self.bowl_pos[:2]
        return np.array([hx, hy, hz, cx, cy, bx, by], dtype=np.float32)

    def step(self, action_pred):
        dx, dy, dz, grip_action = action_pred
        self.step_count += 1

        self.eef_pos[0] += float(dx)
        self.eef_pos[1] += float(dy)
        self.eef_pos[2] += float(dz)

        self.gripper_state = 1.0 if float(grip_action) > 0.5 else 0.0

        # Grasp check
        dist_to_cube = np.linalg.norm(self.eef_pos - self.cube_pos)
        if self.gripper_state > 0.5 and dist_to_cube < 0.08:
            self.cube_held = True

        # Carry cube
        if self.cube_held:
            self.cube_pos = self.eef_pos.copy() - np.array([0, 0, 0.03])

        # Release check
        dist_to_bowl = np.linalg.norm(self.cube_pos[:2] - self.bowl_pos[:2])
        if self.gripper_state < 0.5 and self.cube_held:
            self.cube_held = False
            if dist_to_bowl < 0.12:
                self.cube_pos = self.bowl_pos.copy()

        success = bool((not self.cube_held) and (
            np.linalg.norm(self.cube_pos[:2] - self.bowl_pos[:2]) < 0.08
        ))
        done = bool(success or (self.step_count >= MAX_STEPS))
        return self.get_observation(), success, done


# ---------------------------------------------------------------------------
# Synthetic Frame Renderer
# ---------------------------------------------------------------------------

def render_workspace_frame(eef_pos, cube_pos, bowl_pos, gripper_state,
                            cube_held=False, img_size=IMG_SIZE):
    """
    Render a top-down synthetic view of the manipulation workspace.
    Produces a 224×224 RGB image that SmolVLA can process as visual input.

    Visual elements (matching egocentric video conventions):
      - Beige table surface
      - Orange bowl (ID 1 place zone)
      - Red/blue cube  (ID 0 pick zone)
      - Green/red EEF  (green=open, red=closed)
      - ArUco-style corner markers at cube and bowl
    """
    frame = np.ones((img_size, img_size, 3), dtype=np.uint8) * 245

    # Table surface
    cv2.rectangle(frame, (5, 5), (img_size - 5, img_size - 5),
                  (210, 190, 160), -1)
    cv2.rectangle(frame, (5, 5), (img_size - 5, img_size - 5),
                  (150, 130, 110), 2)

    def ws2px(x, y):
        """Normalised workspace coords → pixel coords."""
        px = int(np.clip(x, 0, 1) * (img_size - 30) + 15)
        py = int(np.clip(y, 0, 1) * (img_size - 30) + 15)
        return (px, py)

    # -- Bowl (orange, ID 1) --
    bp = ws2px(bowl_pos[0], bowl_pos[1])
    cv2.circle(frame, bp, 22, (20, 140, 255), -1)
    cv2.circle(frame, bp, 22, (10, 80, 200), 2)
    cv2.circle(frame, bp, 10, (30, 100, 200), 2)  # inner ring
    cv2.putText(frame, "1", (bp[0] - 4, bp[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -- Cube (red/blue, ID 0) --
    cp = ws2px(cube_pos[0], cube_pos[1])
    cube_col = (50, 50, 220) if not cube_held else (50, 220, 50)
    cv2.rectangle(frame, (cp[0] - 12, cp[1] - 12),
                  (cp[0] + 12, cp[1] + 12), cube_col, -1)
    cv2.rectangle(frame, (cp[0] - 12, cp[1] - 12),
                  (cp[0] + 12, cp[1] + 12), (0, 0, 0), 1)
    cv2.putText(frame, "0", (cp[0] - 4, cp[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -- End-Effector --
    ep = ws2px(eef_pos[0], eef_pos[1])
    eef_col   = (0, 40, 200)  if gripper_state > 0.5 else (0, 180, 20)
    grip_label = "G" if gripper_state > 0.5 else "O"
    cv2.circle(frame, ep, 14, eef_col, -1)
    cv2.circle(frame, ep, 14, (0, 0, 0), 1)
    cv2.putText(frame, grip_label, (ep[0] - 5, ep[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # -- HUD --
    cv2.putText(frame, f"EEF ({eef_pos[0]:.2f},{eef_pos[1]:.2f})",
                (8, img_size - 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (60, 60, 60), 1)
    cv2.putText(frame, f"Grip: {'CLOSED' if gripper_state > 0.5 else 'OPEN'}",
                (8, img_size - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (60, 60, 60), 1)

    return frame  # BGR uint8


def frame_to_tensor(frame_bgr, device):
    """Convert OpenCV BGR frame to [1, 3, H, W] float32 tensor (0-1 range)."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Policy loaders
# ---------------------------------------------------------------------------

def load_mlp(checkpoint="policy.pt"):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"'{checkpoint}' not found — run 2_train_mlp.py first."
        )
    model = TinyPolicy(input_dim=7, hidden_dim=128, output_dim=4)
    model.load_state_dict(torch.load(checkpoint, weights_only=True,
                                     map_location="cpu"))
    model.eval()
    print(f"✅  TinyPolicy MLP loaded from '{checkpoint}'")
    return model


def load_smolvla(checkpoint=SMOLVLA_CHECKPOINT, device="cpu"):
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(
            f"SmolVLA checkpoint not found: {checkpoint}\n"
            "Run the Colab notebook first."
        )
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    print(f"✅  SmolVLA loaded from '{checkpoint}' on {device.upper()}")
    return policy


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

def run_rollout_mlp(env, model, verbose=True):
    """Run one episode with TinyPolicy MLP."""
    obs    = env.reset()
    traj_x, traj_y, traj_z, grippers = [], [], [], []
    success = False

    for step in range(MAX_STEPS):
        state = env.get_state_vector()
        traj_x.append(state[0]); traj_y.append(state[1])
        traj_z.append(state[2]); grippers.append(obs["gripper"])

        state_t = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            raw = model(state_t).squeeze().numpy()

        dx, dy, dz = raw[0], raw[1], raw[2]
        grip_prob  = float(1 / (1 + np.exp(-raw[3])))
        obs, success, done = env.step((dx, dy, dz, grip_prob))

        if verbose and (step % 20 == 0 or done):
            print(f"  [MLP] Step {step:03d} | EEF ({state[0]:.3f},{state[1]:.3f},{state[2]:.3f}) "
                  f"| Grip {'CLOSED' if grip_prob > 0.5 else 'OPEN ':6s} "
                  f"| Cube held: {env.cube_held}")
        if done:
            break

    return success, env.step_count, traj_x, traj_y, traj_z, grippers


def run_rollout_smolvla(env, policy, device, verbose=True):
    """Run one episode with SmolVLA, feeding synthetic rendered frames."""
    obs     = env.reset()
    traj_x, traj_y, traj_z, grippers = [], [], [], []
    success = False

    for step in range(MAX_STEPS):
        state = env.get_state_vector()
        traj_x.append(state[0]); traj_y.append(state[1])
        traj_z.append(state[2]); grippers.append(obs["gripper"])

        # Render synthetic frame
        frame    = render_workspace_frame(
            state[:3], state[3:5], state[5:7],
            obs["gripper"], env.cube_held
        )
        img_t    = frame_to_tensor(frame, device)
        state_t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        observation = {
            "observation.images.phone": img_t,
            "observation.state":        state_t,
            "task":                     [TASK_DESCRIPTION],
        }
        with torch.no_grad():
            action = policy.select_action(observation).squeeze(0).cpu().numpy()

        dx, dy, dz   = action[0], action[1], action[2]
        grip_prob    = float(1 / (1 + np.exp(-action[3])))
        obs, success, done = env.step((dx, dy, dz, grip_prob))

        if verbose and (step % 20 == 0 or done):
            print(f"  [VLA] Step {step:03d} | EEF ({state[0]:.3f},{state[1]:.3f},{state[2]:.3f}) "
                  f"| Grip {'CLOSED' if grip_prob > 0.5 else 'OPEN ':6s} "
                  f"| Cube held: {env.cube_held}")
        if done:
            break

    return success, env.step_count, traj_x, traj_y, traj_z, grippers


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(traj_x, traj_y, grippers, env, title, save_path):
    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    # Table background
    ax.set_facecolor("#f5f0e8")

    ax.plot(traj_x, traj_y, "b-", linewidth=2, label="EEF trajectory", zorder=3)
    ax.scatter([traj_x[0]], [traj_y[0]], c="green", s=120,
               zorder=5, label="Start", marker="o")
    ax.scatter([traj_x[-1]], [traj_y[-1]], c="purple", s=120,
               zorder=5, label="End", marker="*")

    ax.scatter(*env.cube_pos_init[:2], c="red", s=200, marker="s",
               zorder=4, label="Cube (ID 0)")
    ax.scatter(*env.bowl_pos_init[:2], c="orange", s=230, marker="o",
               zorder=4, label="Bowl (ID 1)")

    # Closed-gripper segments in red
    for i in range(len(traj_x) - 1):
        if grippers[i] > 0.5:
            ax.plot(traj_x[i:i+2], traj_y[i:i+2], "r-", linewidth=3.5, alpha=0.8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Workspace X (normalised)")
    ax.set_ylabel("Workspace Y (normalised)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Trajectory plot → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LIBERO-style closed-loop rollout")
    p.add_argument("--policy",     choices=["mlp", "smolvla", "both"],
                   default="mlp")
    p.add_argument("--trials",     type=int, default=1,
                   help="Number of rollout trials per policy")
    p.add_argument("--checkpoint", default="policy.pt",
                   help="TinyPolicy checkpoint (default: policy.pt)")
    p.add_argument("--smolvla",    default=SMOLVLA_CHECKPOINT,
                   help="SmolVLA checkpoint directory")
    p.add_argument("--cpu",        action="store_true",
                   help="Force CPU for SmolVLA")
    p.add_argument("--save-frames", action="store_true",
                   help="Save synthetic frames to scratch/ during VLA rollout")
    return p.parse_args()


def run_policy(policy_name, args, env):
    """Run N trials for one policy. Returns list of result dicts."""
    device  = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    results = []

    if policy_name == "mlp":
        model = load_mlp(args.checkpoint)
        for t in range(args.trials):
            print(f"\n--- MLP Trial {t+1}/{args.trials} ---")
            t0      = time.time()
            success, steps, tx, ty, tz, grips = run_rollout_mlp(env, model)
            elapsed = time.time() - t0
            results.append({"trial": int(t+1), "success": bool(success),
                             "steps": int(steps), "time_s": float(round(elapsed, 2))})
            print(f"  Result: {'✅ SUCCESS' if success else '❌ TIMEOUT'} "
                  f"in {steps} steps ({elapsed:.1f}s)")

            if t == 0:
                os.makedirs("images", exist_ok=True)
                plot_trajectory(tx, ty, grips, env,
                    f"TinyPolicy MLP — Trial 1 — {'SUCCESS' if success else 'TIMEOUT'}",
                    "images/rollout_trajectory.png")

    else:  # smolvla
        policy = load_smolvla(args.smolvla, device)
        os.makedirs("scratch", exist_ok=True)
        for t in range(args.trials):
            print(f"\n--- SmolVLA Trial {t+1}/{args.trials} ---")
            t0      = time.time()
            success, steps, tx, ty, tz, grips = run_rollout_smolvla(
                env, policy, device
            )
            elapsed = time.time() - t0
            results.append({"trial": int(t+1), "success": bool(success),
                             "steps": int(steps), "time_s": float(round(elapsed, 2))})
            print(f"  Result: {'✅ SUCCESS' if success else '❌ TIMEOUT'} "
                  f"in {steps} steps ({elapsed:.1f}s)")

            if t == 0:
                os.makedirs("images", exist_ok=True)
                plot_trajectory(tx, ty, grips, env,
                    f"SmolVLA Policy — Trial 1 — {'SUCCESS' if success else 'TIMEOUT'}",
                    "images/rollout_smolvla_trajectory.png")

    return results


def main():
    args = parse_args()
    env  = SimulatedFrankaWorkspace()

    all_results = {}

    if args.policy in ("mlp", "both"):
        all_results["mlp"] = run_policy("mlp", args, env)

    if args.policy in ("smolvla", "both"):
        all_results["smolvla"] = run_policy("smolvla", args, env)

    # Summary
    print("\n" + "=" * 60)
    print("ROLLOUT SUMMARY")
    print("=" * 60)
    for pol, res in all_results.items():
        successes = sum(r["success"] for r in res)
        total     = len(res)
        avg_steps = np.mean([r["steps"] for r in res])
        print(f"  {pol.upper():10s}  Success: {successes}/{total} "
              f"({100*successes/total:.0f}%)  Avg steps: {avg_steps:.1f}")

    # Save results
    out = {"task": TASK_DESCRIPTION, "max_steps": MAX_STEPS, "results": all_results}
    with open("rollout_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved → rollout_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
