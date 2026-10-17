"""
agents/evaluation_agent.py
==========================
Evaluation Agent — orchestrates multi-trial policy benchmarking and
generates structured evaluation reports.

Responsibilities:
  1. Load all available policies (MLP, SmolVLA fine-tuned)
  2. Run N simulation trials per policy
  3. Compute success rate, efficiency, inference speed
  4. Save metrics to eval_agent_report.json
  5. Generate comparison plots

Usage
-----
  from agents.evaluation_agent import EvaluationAgent
  agent = EvaluationAgent(n_trials=10)
  report = agent.run()
"""

import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from pathlib import Path

# Add parent dir to path so we can import from project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from _libero_sim import (
    SimulatedFrankaWorkspace,
    run_rollout_mlp,
    run_rollout_smolvla,
    TASK_DESCRIPTION,
    MAX_STEPS,
)
from model import TinyPolicy


FINETUNED_CHECKPOINT = (
    "smolvla_checkpoint-20261010T074618Z-1-001"
    "/smolvla_checkpoint/checkpoints/010000/pretrained_model"
)


class EvaluationAgent:
    """
    Automated multi-trial policy evaluation agent.

    Parameters
    ----------
    n_trials     : int    Number of trials per policy
    policies     : list   Which policies to evaluate: ['mlp', 'smolvla']
    output_json  : str    Path to save evaluation report
    output_plot  : str    Path to save comparison plot
    device       : str    'auto', 'cpu', or 'cuda'
    """

    def __init__(
        self,
        n_trials=10,
        policies=None,
        output_json="eval_agent_report.json",
        output_plot="eval_agent_comparison.png",
        device="auto",
        mlp_checkpoint="policy.pt",
        smolvla_checkpoint=FINETUNED_CHECKPOINT,
    ):
        self.n_trials           = n_trials
        self.policies           = policies or ["mlp", "smolvla"]
        self.output_json        = output_json
        self.output_plot        = output_plot
        self.mlp_checkpoint     = mlp_checkpoint
        self.smolvla_checkpoint = smolvla_checkpoint

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    def _load_mlp(self):
        model = TinyPolicy(input_dim=7, hidden_dim=128, output_dim=4)
        model.load_state_dict(torch.load(self.mlp_checkpoint,
                                         weights_only=True,
                                         map_location="cpu"))
        model.eval()
        print(f"  [EvalAgent] TinyPolicy MLP loaded from '{self.mlp_checkpoint}'")
        return model

    def _load_smolvla(self):
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(self.smolvla_checkpoint)
        policy.to(self.device)
        policy.eval()
        print(f"  [EvalAgent] SmolVLA loaded on {self.device.upper()}")
        return policy

    # ------------------------------------------------------------------
    def _run_trials(self, policy_key, policy_obj, env):
        results = []
        for t in range(self.n_trials):
            t_start = time.perf_counter()
            if policy_key == "mlp":
                success, steps, *_ = run_rollout_mlp(env, policy_obj, verbose=False)
            else:
                success, steps, *_ = run_rollout_smolvla(
                    env, policy_obj, self.device, verbose=False
                )
            elapsed     = time.perf_counter() - t_start
            ms_per_step = elapsed / max(steps, 1) * 1000

            results.append({
                "trial":       t + 1,
                "success":     bool(success),
                "steps":       steps,
                "time_s":      round(elapsed, 3),
                "ms_per_step": round(ms_per_step, 2),
            })
            status = "✅" if success else "❌"
            print(f"    Trial {t+1:2d}/{self.n_trials}  {status}  "
                  f"steps={steps:3d}  {elapsed:.2f}s")
        return results

    # ------------------------------------------------------------------
    def _compute_stats(self, results):
        n          = len(results)
        n_ok       = sum(r["success"] for r in results)
        steps_ok   = [r["steps"] for r in results if r["success"]]
        steps_all  = [r["steps"] for r in results]
        ms_list    = [r["ms_per_step"] for r in results]
        return {
            "n_trials":        n,
            "n_success":       n_ok,
            "success_rate":    round(n_ok / n, 3),
            "mean_steps":      round(float(np.mean(steps_all)), 1),
            "mean_steps_ok":   round(float(np.mean(steps_ok)), 1) if steps_ok else None,
            "mean_ms_step":    round(float(np.mean(ms_list)), 2),
            "min_steps_ok":    int(min(steps_ok)) if steps_ok else None,
        }

    # ------------------------------------------------------------------
    def _plot(self, all_stats, all_results):
        names  = list(all_stats.keys())
        colors = ["#2196F3", "#4CAF50", "#FF9800"][:len(names)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("EvaluationAgent — Policy Benchmark", fontsize=13, fontweight="bold")

        # Bar: success rate
        ax = axes[0]
        rates = [all_stats[n]["success_rate"] * 100 for n in names]
        bars  = ax.bar(names, rates, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Success Rate  (n={self.n_trials} trials each)")
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{rate:.0f}%", ha="center", fontweight="bold")

        # Scatter: per-trial timeline
        ax = axes[1]
        for yi, (name, col) in enumerate(zip(names, colors)):
            for ri, res in enumerate(all_results[name]):
                fc = "#4CAF50" if res["success"] else "#F44336"
                ax.scatter(ri+1, yi, s=220, c=fc,
                           edgecolors="black", linewidths=0.5, zorder=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Trial #")
        ax.set_title("Per-Trial Outcomes")
        ax.set_xlim(0.3, self.n_trials + 0.7)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        ok   = mpatches.Patch(color="#4CAF50", label="Success")
        fail = mpatches.Patch(color="#F44336", label="Failure")
        ax.legend(handles=[ok, fail], fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [EvalAgent] Plot saved → {self.output_plot}")

    # ------------------------------------------------------------------
    def run(self, verbose=True):
        """Run evaluation and return the full report dict."""
        env = SimulatedFrankaWorkspace()
        all_results = {}
        all_stats   = {}

        if verbose:
            print(f"\n[EvaluationAgent] Starting benchmark")
            print(f"  Policies : {self.policies}")
            print(f"  Trials   : {self.n_trials} each")
            print(f"  Task     : {TASK_DESCRIPTION}")
            print(f"  Device   : {self.device.upper()}")

        for policy_key in self.policies:
            label = "TinyPolicy MLP" if policy_key == "mlp" else "SmolVLA (Fine-tuned)"
            if verbose:
                print(f"\n  --- {label} ---")

            try:
                if policy_key == "mlp":
                    if not os.path.exists(self.mlp_checkpoint):
                        print(f"  ⚠️  {self.mlp_checkpoint} not found — skipping.")
                        continue
                    pol = self._load_mlp()
                else:
                    if not os.path.isdir(self.smolvla_checkpoint):
                        print(f"  ⚠️  SmolVLA checkpoint not found — skipping.")
                        continue
                    pol = self._load_smolvla()

                results = self._run_trials(policy_key, pol, env)
                stats   = self._compute_stats(results)
                all_results[label] = results
                all_stats[label]   = stats

                if verbose:
                    print(f"  → SR={stats['success_rate']*100:.0f}%  "
                          f"mean_steps={stats['mean_steps']}  "
                          f"{stats['mean_ms_step']} ms/step")

            except Exception as e:
                print(f"  ❌  {label} failed: {e}")

        # Save report
        report = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "task":         TASK_DESCRIPTION,
            "n_trials":     self.n_trials,
            "max_steps":    MAX_STEPS,
            "device":       self.device,
            "stats":        all_stats,
            "results":      all_results,
        }
        with open(self.output_json, "w") as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"\n  [EvalAgent] Report saved → {self.output_json}")

        # Plot
        if len(all_stats) >= 2:
            self._plot(all_stats, all_results)
        elif len(all_stats) == 1:
            print("  [EvalAgent] Need ≥2 policies for comparison plot.")

        return report


if __name__ == "__main__":
    agent = EvaluationAgent(n_trials=10)
    agent.run()
