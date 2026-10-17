"""
5_evaluate_policy.py
====================
Evaluates the trained BC policy (policy.pt) on held-out validation episodes
from dataset.csv. Produces:
  - Per-episode position MSE and gripper accuracy
  - eval_results.json  (detailed metrics)
  - eval_trajectory_plots.png  (predicted vs ground-truth trajectories)
"""

import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import TinyPolicy

DATASET_CSV  = "dataset.csv"
CHECKPOINT   = "policy.pt"
VAL_FRACTION = 0.2       # same split used during training
SEED         = 42

STATE_COLS  = ["hand_x", "hand_y", "hand_z", "cube_x", "cube_y", "bowl_x", "bowl_y"]
ACTION_COLS = ["d_x",    "d_y",    "d_z",    "d_gripper"]


def run_evaluation():
    df       = pd.read_csv(DATASET_CSV)
    episodes = sorted(df["episode_id"].unique())

    # Reproduce same train/val split used in 2_train_mlp.py
    rng      = np.random.default_rng(SEED)
    val_eps  = set(rng.choice(episodes, size=max(1, int(len(episodes) * VAL_FRACTION)), replace=False))
    val_df   = df[df["episode_id"].isin(val_eps)].reset_index(drop=True)

    print(f"Evaluation on {len(val_eps)} validation episodes ({len(val_df)} timesteps)")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TinyPolicy(input_dim=7, hidden_dim=128, output_dim=4).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location=device))
    model.eval()

    results      = []
    ep_results   = []

    val_ep_list  = sorted(val_eps)
    n_plot_cols  = min(3, len(val_ep_list))
    fig          = plt.figure(figsize=(6 * n_plot_cols, 4 * ((len(val_ep_list) + n_plot_cols - 1) // n_plot_cols)))
    plot_idx     = 1

    for ep_id in val_ep_list:
        ep_df = val_df[val_df["episode_id"] == ep_id].reset_index(drop=True)
        X  = torch.tensor(ep_df[STATE_COLS].values,  dtype=torch.float32).to(device)
        Y  = ep_df[ACTION_COLS].values.astype(np.float32)

        with torch.no_grad():
            preds = model(X).cpu().numpy()

        pos_mse  = float(np.mean((preds[:, :3] - Y[:, :3]) ** 2))
        grip_gt  = (Y[:, 3] > 0.5).astype(int)
        grip_pr  = (1 / (1 + np.exp(-preds[:, 3])) > 0.5).astype(int)
        grip_acc = float(np.mean(grip_gt == grip_pr))

        results.append({"episode_id": int(ep_id), "pos_mse": pos_mse, "gripper_acc": grip_acc})
        ep_results.append({"episode_id": int(ep_id), "n_frames": len(ep_df), "pos_mse": pos_mse, "gripper_acc": grip_acc})
        print(f"  Ep {ep_id:02d} | Frames: {len(ep_df):3d} | Pos MSE: {pos_mse:.6f} | Grip Acc: {grip_acc:.1%}")

        # Plot predicted vs GT trajectory (X-Y plane)
        ax = fig.add_subplot((len(val_ep_list) + n_plot_cols - 1) // n_plot_cols, n_plot_cols, plot_idx)
        cumx_gt  = np.cumsum(Y[:, 0]);     cumy_gt  = np.cumsum(Y[:, 1])
        cumx_pr  = np.cumsum(preds[:, 0]); cumy_pr  = np.cumsum(preds[:, 1])
        ax.plot(cumx_gt,  cumy_gt,  "b-",  linewidth=1.5, label="Ground Truth", alpha=0.9)
        ax.plot(cumx_pr,  cumy_pr,  "r--", linewidth=1.5, label="Predicted",    alpha=0.9)
        ax.set_title(f"Ep {ep_id}  |  MSE={pos_mse:.5f}  Grip={grip_acc:.0%}", fontsize=9)
        ax.set_xlabel("Cumul. ΔX"); ax.set_ylabel("Cumul. ΔY")
        ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.5)
        plot_idx += 1

    plt.suptitle("Behavioral Cloning Policy: Predicted vs Ground-Truth Manipulation Trajectories", fontsize=11, y=1.01)
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/eval_trajectory_plots.png", dpi=150, bbox_inches="tight")
    print("Trajectory plots saved to images/eval_trajectory_plots.png")

    # Aggregate metrics
    agg = {
        "mean_pos_mse":      float(np.mean([r["pos_mse"]     for r in results])),
        "median_pos_mse":    float(np.median([r["pos_mse"]   for r in results])),
        "mean_gripper_acc":  float(np.mean([r["gripper_acc"] for r in results])),
        "n_val_episodes":    len(results),
        "n_val_timesteps":   len(val_df),
        "per_episode":       ep_results,
    }

    with open("eval_results.json", "w") as f:
        json.dump(agg, f, indent=2)

    print("\n======= EVALUATION SUMMARY =======")
    print(f"Val Episodes:    {agg['n_val_episodes']}")
    print(f"Val Timesteps:   {agg['n_val_timesteps']}")
    print(f"Mean Pos MSE:    {agg['mean_pos_mse']:.6f}")
    print(f"Mean Grip Acc:   {agg['mean_gripper_acc']:.1%}")
    print("Results saved to eval_results.json")
    print("==================================")


if __name__ == "__main__":
    run_evaluation()
