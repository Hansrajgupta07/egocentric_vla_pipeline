"""
agents/doc_agent.py
===================
Documentation Agent — automatically generates a structured experiment report
from all available output files in the project directory.

Reads:
  - eval_results.json          (TinyPolicy offline evaluation)
  - training_history.json      (MLP training curves)
  - policy_comparison.json     (multi-policy benchmark)
  - eval_agent_report.json     (EvaluationAgent output)
  - dataset_meta.json          (DatasetAgent output)

Writes:
  - EXPERIMENT_REPORT.md       (full technical report)

Usage
-----
  from agents.doc_agent import DocAgent
  agent = DocAgent()
  agent.run()
"""

import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path


class DocAgent:
    """
    Automated documentation agent that assembles a full experiment report
    from all JSON/image artifacts in the project.
    """

    PROJECT_NAME = "Egocentric VLA Pipeline"
    TASK         = "Pick up the Rubik's cube and place it inside the bowl"

    def __init__(self, output_path="EXPERIMENT_REPORT.md"):
        self.output_path = output_path

    # ------------------------------------------------------------------
    def _load_json(self, path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    # ------------------------------------------------------------------
    def _section_dataset(self, meta):
        if meta is None:
            return "## Dataset\n\n> `dataset_meta.json` not found — run DatasetAgent first.\n\n"
        lines = [
            "## Dataset\n",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Total videos | {meta.get('total_videos', '—')} |",
            f"| Valid (pass QA) | {meta.get('valid_videos', '—')} |",
            f"| Rejected | {meta.get('invalid_videos', '—')} |",
            f"| Train episodes | {meta.get('train_episodes', '—')} |",
            f"| Val episodes | {meta.get('val_episodes', '—')} |",
            f"| QA threshold (hand frames) | {meta.get('qa_threshold', 35)} |",
            "",
        ]
        rejected = meta.get("rejected", [])
        if rejected:
            lines.append(f"**Rejected demonstrations:** {', '.join(rejected)}\n")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    def _section_training(self, history):
        if history is None:
            return "## TinyPolicy MLP Training\n\n> `training_history.json` not found.\n\n"
        train_losses = history.get("train_loss", [])
        val_losses   = history.get("val_loss", [])
        val_mse      = history.get("val_pos_mse", [])
        n_epochs     = len(train_losses)

        final_train = train_losses[-1] if train_losses else "—"
        final_val   = val_losses[-1]   if val_losses   else "—"
        best_val    = min(val_losses)   if val_losses   else "—"
        best_epoch  = val_losses.index(min(val_losses)) + 1 if val_losses else "—"
        final_mse   = val_mse[-1]      if val_mse      else "—"

        lines = [
            "## TinyPolicy MLP Training\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Epochs | {n_epochs} |",
            f"| Final train loss | {final_train:.5f} |",
            f"| Final val loss | {final_val:.5f} |",
            f"| Best val loss | {best_val:.5f} (epoch {best_epoch}) |",
            f"| Final val pos MSE | {final_mse:.6f} |",
            "",
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    def _section_offline_eval(self, eval_data):
        if eval_data is None:
            return "## Offline Evaluation (TinyPolicy)\n\n> `eval_results.json` not found.\n\n"
        lines = [
            "## Offline Evaluation — TinyPolicy MLP\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Val episodes | {eval_data.get('n_val_episodes', '—')} |",
            f"| Val timesteps | {eval_data.get('n_val_timesteps', '—')} |",
            f"| Mean position MSE | {eval_data.get('mean_pos_mse', 0):.6f} |",
            f"| Median position MSE | {eval_data.get('median_pos_mse', 0):.6f} |",
            f"| Mean gripper accuracy | {eval_data.get('mean_gripper_acc', 0)*100:.1f}% |",
            "",
            "### Per-Episode Results\n",
            "| Episode | Frames | Pos MSE | Gripper Acc |",
            "|---------|--------|---------|------------|",
        ]
        for ep in eval_data.get("per_episode", []):
            lines.append(
                f"| {ep['episode_id']} | {ep['n_frames']} | "
                f"{ep['pos_mse']:.6f} | {ep['gripper_acc']*100:.0f}% |"
            )
        return "\n".join(lines) + "\n\n"

    # ------------------------------------------------------------------
    def _section_simulation(self, comparison):
        if comparison is None:
            # Try rollout_results.json as fallback
            comparison = self._load_json("rollout_results.json")
        if comparison is None:
            return ("## Simulation Evaluation\n\n"
                    "> Run `python 3_run_libero.py --policy both` or "
                    "`python 8_compare_policies.py` to generate results.\n\n")

        stats = comparison.get("stats", {})
        lines = [
            "## Simulation Evaluation — Closed-Loop Rollout\n",
            f"- **Environment:** SimulatedFrankaWorkspace (normalised coordinates)",
            f"- **Task:** {comparison.get('task', self.TASK)}",
            f"- **Max steps per trial:** {comparison.get('max_steps', 120)}",
            "",
            "| Policy | Trials | Success Rate | Mean Steps | ms/step |",
            "|--------|--------|-------------|-----------|---------|",
        ]
        for name, s in stats.items():
            lines.append(
                f"| {name} | {s.get('n_trials','—')} | "
                f"{s.get('success_rate',0)*100:.0f}% "
                f"({s.get('n_success','—')}/{s.get('n_trials','—')}) | "
                f"{s.get('mean_steps','—')} | "
                f"{s.get('mean_ms_step','—')} |"
            )
        lines.append("")

        if os.path.exists("policy_comparison.png"):
            lines.append("![Policy Comparison](policy_comparison.png)\n")
        if os.path.exists("eval_agent_comparison.png"):
            lines.append("![Evaluation Agent Comparison](eval_agent_comparison.png)\n")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    def _section_pipeline(self):
        return """## Pipeline Architecture

```
Phone Egocentric Video (raw_videos/)
         │
         ▼
[DatasetAgent] ──── QA + train/val split ──► dataset_meta.json
         │
         ▼
[AnnotationAgent] ─ hand + ArUco + phase ──► annotations/
         │
         ▼
1_extract_features.py ─────────────────────► dataset.csv
         │
         ▼
2_train_mlp.py ────────────────────────────► policy.pt (TinyPolicy MLP)
         │
         ▼
4_build_lerobot_dataset.py ────────────────► lerobot_dataset/ (LeRobot v2.0)
         │
         ▼
[Colab] notebooks/colab_finetune_smolvla.py► smolvla_checkpoint/ (450M VLA)
         │
         ├─── 3_run_libero.py (--policy mlp)     ► rollout_trajectory.png
         ├─── 3_run_libero.py (--policy smolvla)  ► rollout_smolvla_trajectory.png
         │
         ▼
5_evaluate_policy.py ──────────────────────► eval_results.json
8_compare_policies.py ─────────────────────► policy_comparison.json + .png
         │
         ▼
[EvaluationAgent] ─────────────────────────► eval_agent_report.json
         │
         ▼
[DocAgent] ────────────────────────────────► EXPERIMENT_REPORT.md  ← this file
```

### Retargeting Strategy

Phone video → monocular 2D observation.  
**ArUco markers** (DICT_4X4_50) placed on cube (ID 0) and bowl (ID 1) provide
reliable 2D localisation without depth sensors.

Human hand trajectory extracted with **MediaPipe HandLandmarker**:
- Wrist landmark → end-effector XY position
- Relative hand depth (z) → approximate Z displacement
- Thumb-index pinch distance → gripper open/close binary signal

These 7-dim states + 4-dim action deltas form the robot demonstration dataset
that is used to:
1. Train a lightweight TinyPolicy MLP baseline (CPU, < 2 min)
2. Fine-tune the pretrained SmolVLA VLA model (Colab GPU, ~45 min / 10k steps)

"""

    # ------------------------------------------------------------------
    def _section_next_steps(self):
        return """## Next Steps

- [ ] Push SmolVLA checkpoint to HuggingFace Hub
- [ ] Install real LIBERO and evaluate in the actual simulator
- [ ] Collect 50+ demonstrations for improved generalisation
- [ ] Add depth estimation (MiDaS) for better Z coordinate
- [ ] Implement action chunking (4-step horizon) in the TinyPolicy
- [ ] Explore RL fine-tuning of the policy in simulation
- [ ] Test on real Franka Panda hardware

"""

    # ------------------------------------------------------------------
    def run(self, verbose=True):
        if verbose:
            print("[DocAgent] Assembling experiment report ...")

        meta       = self._load_json("dataset_meta.json")
        history    = self._load_json("training_history.json")
        eval_data  = self._load_json("eval_results.json")
        comparison = self._load_json("policy_comparison.json") or \
                     self._load_json("eval_agent_report.json")

        lines = [
            f"# Experiment Report — {self.PROJECT_NAME}",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC  ",
            f"**Task:** {self.TASK}  ",
            f"**Model:** SmolVLA 450M (HuggingFaceTB/SmolVLM2-500M-Video-Instruct)  ",
            f"**Fine-tuning:** 10,000 steps on egocentric phone demonstration dataset  ",
            "",
            "---",
            "",
        ]

        lines.append(self._section_pipeline())
        lines.append(self._section_dataset(meta))
        lines.append(self._section_training(history))
        lines.append(self._section_offline_eval(eval_data))
        lines.append(self._section_simulation(comparison))
        lines.append(self._section_next_steps())

        # Figures
        lines.append("## Figures\n")
        figs = [
            ("rollout_trajectory.png",        "TinyPolicy MLP closed-loop rollout"),
            ("rollout_smolvla_trajectory.png", "SmolVLA closed-loop rollout"),
            ("eval_trajectory_plots.png",      "Offline evaluation: predicted vs GT"),
            ("policy_comparison.png",          "Policy comparison benchmark"),
        ]
        for fname, caption in figs:
            target_path = os.path.join("images", fname) if os.path.exists(os.path.join("images", fname)) else fname
            if os.path.exists(target_path):
                lines.append(f"### {caption}\n")
                lines.append(f"![{caption}]({target_path})\n")

        report_text = "\n".join(lines)
        Path(self.output_path).write_text(report_text)

        if verbose:
            print(f"[DocAgent] Report saved → {self.output_path}")
            print(f"  Sections: Dataset, Training, Offline Eval, Simulation, Pipeline")

        return report_text


if __name__ == "__main__":
    agent = DocAgent()
    agent.run()
