"""
============================================================
 SmolVLA Fine-Tuning on Egocentric Pick-and-Place Dataset
 Google Colab Notebook  |  GPU: A100 / T4  |  ~45 minutes
============================================================

HOW TO USE:
  1. Upload the entire 'lerobot_dataset/' folder to your Google Drive.
  2. Open a new Google Colab notebook (colab.research.google.com).
  3. Set Runtime -> Change Runtime Type -> A100 GPU (free tier).
  4. Copy-paste each cell block below into sequential Colab cells.
  5. Click "Run All".

This file is structured as Python code with ### CELL N: markers
so you can easily split it into Colab cells.
"""

### CELL 1: Mount Google Drive & Clone LeRobot ###
"""
!pip install -q lerobot
from google.colab import drive
drive.mount('/content/drive')

import shutil, os

# --- EDIT THIS to match your Drive folder ---
DRIVE_DATASET_PATH = "/content/drive/MyDrive/egocentric_vla/lerobot_dataset"
LOCAL_DATASET_PATH = "/content/lerobot_dataset"

if os.path.exists(LOCAL_DATASET_PATH):
    shutil.rmtree(LOCAL_DATASET_PATH)
shutil.copytree(DRIVE_DATASET_PATH, LOCAL_DATASET_PATH)
print("Dataset copied to:", LOCAL_DATASET_PATH)
"""

### CELL 2: Verify Dataset & Patch Stats ###
"""
import json
from pathlib import Path

dataset_root = Path(LOCAL_DATASET_PATH)
info_path = dataset_root / "meta" / "info.json"
stats_path = dataset_root / "meta" / "stats.json"
episodes_path = dataset_root / "meta" / "episodes.jsonl"

info = json.loads(info_path.read_text())
print("=== Dataset Info ===")
print(f"  Total Episodes:  {info['total_episodes']}")
print(f"  Total Frames:    {info['total_frames']}")
print(f"  FPS:             {info['fps']}")
print(f"  Camera Key:      {list(k for k in info['features'] if 'image' in k)}")
print(f"  Action Dim:      {info['features']['action']['shape']}")
print(f"  State Dim:       {info['features']['observation.state']['shape']}")

episodes = [json.loads(l) for l in episodes_path.read_text().splitlines()]
print(f"  Episodes loaded: {len(episodes)}")

# Patch stats.json with ImageNet stats for camera if missing
if stats_path.exists():
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    # Check what the camera key is
    cam_key = [k for k in info['features'] if 'image' in k][0]
    
    if cam_key not in stats:
        stats[cam_key] = {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
            "min":  [0.0, 0.0, 0.0],
            "max":  [1.0, 1.0, 1.0],
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  [Auto-Patch] Added ImageNet stats for '{cam_key}' to stats.json!")
    else:
        print(f"  [Stats] Camera '{cam_key}' stats already present.")
"""

### CELL 3: Configure SmolVLA Training ###
"""
import os

# HuggingFace repo to push the trained model to (change to your username)
HF_REPO_ID = "your-username/egocentric-pick-place-smolvla"

training_config = {
    "policy_type":      "smolvla",
    "dataset_root":     LOCAL_DATASET_PATH,
    "output_dir":       "/content/smolvla_checkpoint",

    # Training hyperparameters
    "steps":            10000,
    "batch_size":       8,
    "lr":               1e-4,
    "lr_warmup_steps":  500,
    "grad_clip_norm":   10.0,
    "weight_decay":     1e-4,
    "eval_freq":        1000,
    "save_freq":        2000,
    "seed":             42,

    # SmolVLA specific
    "chunk_size":       4,          # action chunk horizon
    "n_action_steps":   4,          # steps per inference
    "camera_names":     ["observation.images.phone"],

    # Data
    "image_size":       [224, 224],
    "fps":              24,
    "task_description": "Pick up the Rubik's cube and place it inside the bowl",
}
print("Training config set. Steps:", training_config["steps"])
"""

### CELL 4: Launch SmolVLA Fine-Tuning ###
"""
from lerobot.scripts.train import train_cli
import subprocess, sys

# Build the training command
cmd = [
    sys.executable, "-m", "lerobot.scripts.train",
    "--policy.type=smolvla",
    f"--dataset.root={training_config['dataset_root']}",
    f"--output_dir={training_config['output_dir']}",
    "--steps=10000",
    "--batch_size=8",
    "--optimizer.lr=1e-4",
    "--scheduler.type=constant_with_warmup",
    "--scheduler.num_warmup_steps=500",
    "--optimizer.grad_clip_norm=10.0",
    "--policy.device=cuda",
    "--policy.use_amp=true",
    "--policy.push_to_hub=false",
    f"--policy.chunk_size={training_config['chunk_size']}",
    f"--policy.n_action_steps={training_config['n_action_steps']}",
]

print("Launching training with command:")
print(" ".join(cmd))
result = subprocess.run(cmd, capture_output=False)
print("\\nTraining complete! Exit code:", result.returncode)
"""

### CELL 5: Plot Training Curves ###
"""
import json, glob
import matplotlib.pyplot as plt

# LeRobot logs to JSON - find the latest run
log_files = sorted(glob.glob(f"{training_config['output_dir']}/**/training_log.json", recursive=True))
if not log_files:
    log_files = sorted(glob.glob(f"{training_config['output_dir']}/**/*.json", recursive=True))

print("Log files found:", log_files)

# Load and parse training metrics
if log_files:
    with open(log_files[-1]) as f:
        logs = json.load(f)

    steps  = [l["step"]  for l in logs if "step" in l]
    losses = [l["loss"]  for l in logs if "loss" in l]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, losses, "b-", linewidth=2)
    plt.title("SmolVLA Fine-Tuning Loss - Egocentric Pick-and-Place")
    plt.xlabel("Training Steps")
    plt.ylabel("Action Prediction Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("/content/training_curve.png", dpi=150)
    plt.show()
    print("Training curve saved!")
"""

### CELL 6: Evaluate Trained SmolVLA Policy ###
"""
import torch
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from pathlib import Path
import numpy as np

checkpoint_dir = Path(training_config["output_dir"])
policy = SmolVLAPolicy.from_pretrained(str(checkpoint_dir))
policy.eval()
print("SmolVLA policy loaded from:", checkpoint_dir)

# Run inference on a single held-out episode frame
import cv2
import json

# Load a sample frame from the dataset
sample_video = sorted(Path(LOCAL_DATASET_PATH).glob("videos/**/*.mp4"))[0]
cap = cv2.VideoCapture(str(sample_video))
ret, frame = cap.read()
cap.release()

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
frame_tensor = frame_tensor.unsqueeze(0)  # [1, C, H, W]

task = training_config["task_description"]
print(f"Running inference for task: '{task}'")

with torch.no_grad():
    action = policy.select_action({
        "observation.images.phone": frame_tensor,
        "task": [task],
    })

print(f"Predicted action: {action.numpy()}")
print(f"  dx={action[0,0]:.4f}, dy={action[0,1]:.4f}, dz={action[0,2]:.4f}, gripper={action[0,3]:.4f}")
"""

### CELL 7: Save Model to HuggingFace Hub ###
"""
from huggingface_hub import notebook_login
notebook_login()   # Enter your HF token

policy.push_to_hub(HF_REPO_ID)
print(f"Model pushed to: https://huggingface.co/{HF_REPO_ID}")
"""

### CELL 8: Download Checkpoint to Drive ###
"""
import shutil
checkpoint_drive = f"/content/drive/MyDrive/egocentric_vla/smolvla_checkpoint"
shutil.copytree(training_config['output_dir'], checkpoint_drive, dirs_exist_ok=True)
print(f"Checkpoint backed up to Google Drive: {checkpoint_drive}")
"""
