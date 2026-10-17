"""
agents/orchestrator.py
======================
Pipeline Orchestrator — runs all agents in the correct order with a single command.

Execution order:
  1. DatasetAgent    — QA all videos, create splits
  2. AnnotationAgent — per-frame annotations + task phases  [optional]
  3. EvaluationAgent — multi-trial policy benchmark
  4. DocAgent        — assemble EXPERIMENT_REPORT.md

Usage
-----
  python agents/orchestrator.py                    # full pipeline
  python agents/orchestrator.py --skip-annotation  # skip annotation (slower step)
  python agents/orchestrator.py --eval-trials 5    # fewer trials
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.dataset_agent    import DatasetAgent
from agents.annotation_agent import AnnotationAgent
from agents.evaluation_agent import EvaluationAgent
from agents.doc_agent        import DocAgent


def banner(msg, width=65):
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def parse_args():
    p = argparse.ArgumentParser(description="Egocentric VLA pipeline orchestrator")
    p.add_argument("--skip-annotation", action="store_true",
                   help="Skip AnnotationAgent (saves time)")
    p.add_argument("--eval-trials",     type=int,   default=10,
                   help="Number of simulation trials per policy (default: 10)")
    p.add_argument("--policies",        nargs="+",  default=["mlp", "smolvla"],
                   help="Policies to evaluate: mlp smolvla")
    p.add_argument("--videos-dir",      default="raw_videos")
    p.add_argument("--cpu",             action="store_true",
                   help="Force CPU for SmolVLA")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cpu" if args.cpu else "auto"
    t0     = time.time()

    print(f"\n🚀  Egocentric VLA Pipeline Orchestrator")
    print(f"    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Policies to evaluate: {args.policies}")
    print(f"    Eval trials: {args.eval_trials}")

    step_times = {}

    # ── Step 1: Dataset QA ──────────────────────────────────────────────
    banner("Step 1/4 — DatasetAgent: Video QA & Splits")
    t1 = time.time()
    dataset_agent = DatasetAgent(videos_dir=args.videos_dir)
    dataset_report = dataset_agent.run()
    step_times["DatasetAgent"] = round(time.time() - t1, 1)

    if not dataset_report:
        print("⚠️  DatasetAgent returned empty report — continuing anyway.")
    else:
        n_valid = dataset_report.get("valid_videos", 0)
        print(f"\n  ✅ DatasetAgent done  ({n_valid} valid demonstrations)")

    # ── Step 2: Annotation ──────────────────────────────────────────────
    if not args.skip_annotation:
        banner("Step 2/4 — AnnotationAgent: Per-Frame Annotations")
        t2 = time.time()
        if os.path.exists("hand_landmarker.task"):
            ann_agent = AnnotationAgent(videos_dir=args.videos_dir)
            ann_agent.run()
            step_times["AnnotationAgent"] = round(time.time() - t2, 1)
            print(f"\n  ✅ AnnotationAgent done")
        else:
            print("  ⚠️  hand_landmarker.task not found — skipping annotation.")
            step_times["AnnotationAgent"] = 0
    else:
        print("\n  [Skipped] AnnotationAgent")
        step_times["AnnotationAgent"] = 0

    # ── Step 3: Evaluation ──────────────────────────────────────────────
    banner("Step 3/4 — EvaluationAgent: Multi-Trial Policy Benchmark")
    t3 = time.time()
    eval_agent = EvaluationAgent(
        n_trials=args.eval_trials,
        policies=args.policies,
        device=device,
    )
    eval_report = eval_agent.run()
    step_times["EvaluationAgent"] = round(time.time() - t3, 1)

    stats = eval_report.get("stats", {})
    print(f"\n  ✅ EvaluationAgent done")
    for pol, s in stats.items():
        print(f"     {pol:30s}  SR={s['success_rate']*100:.0f}%  "
              f"steps={s['mean_steps']}")

    # ── Step 4: Documentation ───────────────────────────────────────────
    banner("Step 4/4 — DocAgent: Generating Experiment Report")
    t4 = time.time()
    doc_agent = DocAgent()
    doc_agent.run()
    step_times["DocAgent"] = round(time.time() - t4, 1)
    print(f"\n  ✅ DocAgent done  →  EXPERIMENT_REPORT.md")

    # ── Final Summary ───────────────────────────────────────────────────
    total_time = round(time.time() - t0, 1)
    banner("✅  Pipeline Complete")
    print(f"\n  Total time : {total_time}s")
    print(f"\n  Agent times:")
    for agent, t in step_times.items():
        print(f"    {agent:20s}  {t}s")

    print(f"\n  Key outputs:")
    outputs = [
        ("dataset_meta.json",       "Dataset QA audit"),
        ("annotations/",            "Per-frame annotations"),
        ("eval_agent_report.json",  "Evaluation metrics"),
        ("eval_agent_comparison.png","Comparison plot"),
        ("EXPERIMENT_REPORT.md",    "Full experiment report"),
    ]
    for fname, desc in outputs:
        exists = os.path.exists(fname)
        marker = "✅" if exists else "⚪"
        print(f"    {marker}  {fname:30s}  {desc}")

    print()


if __name__ == "__main__":
    main()
