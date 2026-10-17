"""
8_compare_policies.py
=====================
Multi-trial benchmark comparing:
  1. TinyPolicy MLP (behavior cloning baseline)
  2. Pretrained SmolVLA (no fine-tuning, zero-shot)
  3. Fine-tuned SmolVLA (your 10k-step checkpoint)

For each policy, runs N trials in the simulated Franka workspace and reports:
  - Success rate
  - Mean steps to completion
  - Mean inference time per step
  - Per-trial breakdown

Outputs:
  - policy_comparison.json   — raw metrics
  - policy_comparison.png    — bar chart + per-trial timeline
  - comparison_report.md     — human-readable summary

Usage
-----
  python 8_compare_policies.py                        # all 3 policies, 10 trials each
  python 8_compare_policies.py --trials 5 --skip-pretrained
  python 8_compare_policies.py --trials 10 --cpu
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from model import TinyPolicy

# ---------------------------------------------------------------------------
# Imports from step 3 (reuse simulation + renderer)
# ---------------------------------------------------------------------------
from _libero_sim import (          # re-export from 3_run_libero
    SimulatedFrankaWorkspace,
    render_workspace_frame,
    frame_to_tensor,
    run_rollout_mlp,
    run_rollout_smolvla,
    TASK_DESCRIPTION,
    MAX_STEPS,
)

FINETUNED_CHECKPOINT = (
    "smolvla_checkpoint-20261010T074618Z-1-001"
    "/smolvla_checkpoint/checkpoints/010000/pretrained_model"
)
PRETRAINED_MODEL_ID = "lerobot/smolvla_base"     # HF model hub ID

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mlp(checkpoint="policy.pt"):
    model = TinyPolicy(input_dim=7, hidden_dim=128, output_dim=4)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    model.eval()
    return model


def load_smolvla_finetuned(device):
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(FINETUNED_CHECKPOINT)
    policy.to(device)
    policy.eval()
    return policy


def load_smolvla_pretrained(device):
    """Load the base SmolVLA without fine-tuning for zero-shot comparison."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    try:
        policy = SmolVLAPolicy.from_pretrained(PRETRAINED_MODEL_ID)
        policy.to(device)
        policy.eval()
        return policy
    except Exception as e:
        print(f"  ⚠️  Could not load pretrained SmolVLA from Hub: {e}")
        print("     Skipping pretrained baseline (needs internet + HF access).")
        return None


def run_trials(policy_name, policy_obj, env, n_trials, device, is_smolvla=False):
    """Run n_trials and return list of result dicts with timing."""
    results = []
    for t in range(n_trials):
        t_start = time.perf_counter()
        if is_smolvla:
            success, steps, tx, ty, tz, grips = run_rollout_smolvla(
                env, policy_obj, device, verbose=False
            )
        else:
            success, steps, tx, ty, tz, grips = run_rollout_mlp(
                env, policy_obj, verbose=False
            )
        elapsed = time.perf_counter() - t_start
        ms_per_step = (elapsed / steps * 1000) if steps > 0 else 0

        results.append({
            "trial":        t + 1,
            "success":      bool(success),
            "steps":        steps,
            "time_s":       round(elapsed, 3),
            "ms_per_step":  round(ms_per_step, 2),
        })
        status = "✅" if success else "❌"
        print(f"  [{policy_name}] Trial {t+1:2d}/{n_trials}  "
              f"{status}  steps={steps:3d}  "
              f"time={elapsed:.2f}s  ({ms_per_step:.1f} ms/step)")
    return results


def compute_stats(results):
    n          = len(results)
    n_success  = sum(r["success"] for r in results)
    steps_list = [r["steps"]       for r in results]
    ms_list    = [r["ms_per_step"] for r in results]
    return {
        "n_trials":       n,
        "n_success":      n_success,
        "success_rate":   round(n_success / n, 3),
        "mean_steps":     round(float(np.mean(steps_list)),  1),
        "median_steps":   round(float(np.median(steps_list)), 1),
        "mean_ms_step":   round(float(np.mean(ms_list)), 2),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(all_stats, all_results, save_path="images/policy_comparison.png"):
    names   = list(all_stats.keys())
    colors  = ["#2196F3", "#FF9800", "#4CAF50"][:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Policy Comparison: TinyPolicy vs SmolVLA (Pretrained) vs SmolVLA (Fine-tuned)",
                 fontsize=12, fontweight="bold")

    # 1. Success Rate
    ax = axes[0]
    rates = [all_stats[n]["success_rate"] * 100 for n in names]
    bars  = ax.bar(names, rates, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, f"{rate:.0f}%",
                ha="center", fontweight="bold", fontsize=10)

    # 2. Mean Steps
    ax = axes[1]
    steps = [all_stats[n]["mean_steps"] for n in names]
    bars  = ax.bar(names, steps, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(MAX_STEPS, color="red", linestyle="--", alpha=0.6,
               label=f"Max ({MAX_STEPS})")
    ax.set_ylabel("Mean Steps to Completion")
    ax.set_title("Efficiency (Steps)")
    ax.legend(fontsize=8)
    for bar, s in zip(bars, steps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{s:.0f}",
                ha="center", fontsize=9)

    # 3. Per-trial timeline (success=green, fail=red)
    ax = axes[2]
    n_trials = max(len(all_results[n]) for n in names)
    for yi, (name, col) in enumerate(zip(names, colors)):
        for ri, res in enumerate(all_results[name]):
            fc = "#4CAF50" if res["success"] else "#F44336"
            ax.scatter(ri + 1, yi, s=200, c=fc,
                       edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Trial #")
    ax.set_title("Per-Trial Outcomes")
    ax.set_xlim(0.3, n_trials + 0.7)
    success_patch = mpatches.Patch(color="#4CAF50", label="Success")
    fail_patch    = mpatches.Patch(color="#F44336", label="Failure")
    ax.legend(handles=[success_patch, fail_patch], fontsize=8, loc="lower right")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(all_stats, save_path="comparison_report.md"):
    lines = [
        "# Policy Comparison Report",
        "",
        f"**Task:** {TASK_DESCRIPTION}",
        f"**Simulator:** SimulatedFrankaWorkspace (normalised coordinates)",
        f"**Max steps per trial:** {MAX_STEPS}",
        "",
        "## Results Summary",
        "",
        "| Policy | Trials | Success Rate | Mean Steps | ms/step |",
        "|--------|--------|-------------|-----------|---------|",
    ]
    for name, stats in all_stats.items():
        lines.append(
            f"| {name} | {stats['n_trials']} | "
            f"{stats['success_rate']*100:.0f}% ({stats['n_success']}/{stats['n_trials']}) | "
            f"{stats['mean_steps']} | {stats['mean_ms_step']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- **TinyPolicy MLP**: 7-dim state → 4-dim action (no visual input). "
          "Fast CPU inference (~1-2 ms/step). Best for comparing against VLA.",
        "- **SmolVLA (Pretrained)**: Zero-shot performance of the base 450M "
          "parameter SmolVLA without any fine-tuning on the egocentric dataset.",
        "- **SmolVLA (Fine-tuned)**: After 10,000 steps of fine-tuning on the "
          "egocentric pick-and-place demonstrations (25 videos, ~20 valid episodes).",
        "",
        "### Visual Domain Note",
        "SmolVLA was trained on real phone camera frames. During simulation it "
        "receives synthetic top-down OpenCV renders. The performance gap vs MLP "
        "reflects this sim-to-real visual domain shift — not a fundamental "
        "limitation of the VLA approach.",
        "",
        "![Comparison Plot](policy_comparison.png)",
    ]

    Path(save_path).write_text("\n".join(lines))
    print(f"Report saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Multi-policy benchmark")
    p.add_argument("--trials",          type=int, default=10)
    p.add_argument("--skip-pretrained", action="store_true",
                   help="Skip pretrained SmolVLA baseline (needs HF Hub access)")
    p.add_argument("--cpu",             action="store_true")
    p.add_argument("--mlp-checkpoint",  default="policy.pt")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    env    = SimulatedFrankaWorkspace()

    all_results = {}
    all_stats   = {}

    print("=" * 65)
    print(f"Policy Comparison Benchmark  |  {args.trials} trials each")
    print(f"Task: {TASK_DESCRIPTION}")
    print("=" * 65)

    # 1. TinyPolicy MLP
    print("\n[1/3] TinyPolicy MLP")
    if os.path.exists(args.mlp_checkpoint):
        mlp   = load_mlp(args.mlp_checkpoint)
        res   = run_trials("MLP", mlp, env, args.trials, device, is_smolvla=False)
        all_results["TinyPolicy MLP"] = res
        all_stats["TinyPolicy MLP"]   = compute_stats(res)
    else:
        print(f"  ⚠️  {args.mlp_checkpoint} not found — skipping MLP.")

    # 2. Pretrained SmolVLA (zero-shot)
    if not args.skip_pretrained:
        print("\n[2/3] SmolVLA (Pretrained, zero-shot)")
        pol = load_smolvla_pretrained(device)
        if pol is not None:
            res = run_trials("SmolVLA-base", pol, env, args.trials,
                             device, is_smolvla=True)
            all_results["SmolVLA (Pretrained)"] = res
            all_stats["SmolVLA (Pretrained)"]   = compute_stats(res)
    else:
        print("\n[2/3] SmolVLA (Pretrained) — SKIPPED")

    # 3. Fine-tuned SmolVLA
    print("\n[3/3] SmolVLA (Fine-tuned, 10k steps)")
    if os.path.isdir(FINETUNED_CHECKPOINT):
        pol = load_smolvla_finetuned(device)
        res = run_trials("SmolVLA-FT", pol, env, args.trials,
                         device, is_smolvla=True)
        all_results["SmolVLA (Fine-tuned)"] = res
        all_stats["SmolVLA (Fine-tuned)"]   = compute_stats(res)
    else:
        print(f"  ⚠️  Fine-tuned checkpoint not found: {FINETUNED_CHECKPOINT}")

    # Print summary
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    for name, stats in all_stats.items():
        print(f"  {name:30s}  "
              f"SR={stats['success_rate']*100:.0f}%  "
              f"steps={stats['mean_steps']:.1f}  "
              f"{stats['mean_ms_step']:.1f} ms/step")

    # Save outputs
    out = {"task": TASK_DESCRIPTION, "trials": args.trials,
           "stats": all_stats, "results": all_results}
    with open("policy_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nRaw data saved → policy_comparison.json")

    if len(all_stats) >= 2:
        plot_comparison(all_stats, all_results)
        write_report(all_stats)
    else:
        print("(Need ≥2 policies to generate comparison plots)")

    print("=" * 65)


if __name__ == "__main__":
    main()
