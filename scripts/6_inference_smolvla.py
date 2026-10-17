"""
6_inference_smolvla.py
======================
Runs inference with the fine-tuned SmolVLA checkpoint on either:
  - A static image file (--source image --path scratch/sample_62.jpg)
  - A video file     (--source video --path raw_videos/demo.mp4)
  - A live webcam    (--source webcam)

The model expects:
  - observation.images.phone : [1, 3, 224, 224] float32 tensor (0-1 range)
  - observation.state        : [1, 7]  float32 tensor
       [hand_x, hand_y, hand_z, cube_x, cube_y, bowl_x, bowl_y]
  - task                     : list[str]

Outputs a 4-dim action vector: [d_x, d_y, d_z, d_gripper_logit]

Usage
-----
  python 6_inference_smolvla.py                          # webcam (default)
  python 6_inference_smolvla.py --source image --path scratch/sample_62.jpg
  python 6_inference_smolvla.py --source video --path raw_videos/demo.mp4
  python 6_inference_smolvla.py --source image --path scratch/sample_62.jpg --no-detect
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "smolvla_checkpoint-20261010T074618Z-1-001"
    "/smolvla_checkpoint/checkpoints/010000/pretrained_model"
)
TASK_DESCRIPTION = "Pick up the Rubik's cube and place it inside the bowl"

IMG_SIZE      = 224          # model input resolution
STATE_DIM     = 7            # [hand_x, hand_y, hand_z, cube_x, cube_y, bowl_x, bowl_y]
ACTION_DIM    = 4            # [d_x, d_y, d_z, d_gripper_logit]

# Default zone coordinates (normalised 0-1) when ArUco markers are not visible
DEFAULT_CUBE  = (0.75, 0.50)
DEFAULT_BOWL  = (0.25, 0.50)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path: str, device: str):
    """Load the fine-tuned SmolVLA policy from a local pretrained_model directory."""
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not os.path.isdir(checkpoint_path):
        print(f"❌  Checkpoint directory not found: {checkpoint_path}")
        print("    Run the Colab notebook (notebooks/colab_finetune_smolvla.py) first.")
        sys.exit(1)

    print(f"Loading SmolVLA policy from:\n  {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    print(f"✅  SmolVLA loaded on {device.upper()}  |  "
          f"state_dim={STATE_DIM}  action_dim={ACTION_DIM}")
    return policy


def setup_detectors():
    """Initialise ArUco + MediaPipe hand detectors (same settings as step 1)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_det  = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    hand_det = mp_vision.HandLandmarker.create_from_options(options)
    return aruco_det, hand_det


def extract_state(frame_bgr, aruco_det, hand_det,
                  prev_state=None):
    """
    Detect hand wrist + ArUco markers in a BGR frame.
    Returns a 7-dim numpy state vector and an annotated copy of the frame.
    Falls back to prev_state (or defaults) if detection fails.
    """
    h, w = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    # --- ArUco ---
    corners, ids, _ = aruco_det.detectMarkers(frame_bgr)
    cube_x, cube_y = DEFAULT_CUBE
    bowl_x, bowl_y = DEFAULT_BOWL

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for corner, mid in zip(corners, ids.flatten()):
            pts  = corner.reshape((4, 2))
            cx   = float(np.mean(pts[:, 0])) / w
            cy   = float(np.mean(pts[:, 1])) / h
            if mid == 0:
                cube_x, cube_y = cx, cy
            elif mid == 1:
                bowl_x, bowl_y = cx, cy

    # --- Hand ---
    rgb       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_res  = hand_det.detect(mp_img)

    hand_x = hand_y = hand_z = 0.5   # centre defaults
    hand_found = False

    if hand_res.hand_landmarks:
        lm        = hand_res.hand_landmarks[0]
        wrist     = lm[0]
        hand_x    = float(wrist.x)
        hand_y    = float(wrist.y)
        hand_z    = float(wrist.z)
        hand_found = True

        # Draw landmarks
        for lmk in lm:
            cv2.circle(annotated,
                       (int(lmk.x * w), int(lmk.y * h)),
                       4, (0, 255, 0), -1)
        cv2.circle(annotated,
                   (int(hand_x * w), int(hand_y * h)),
                   8, (0, 0, 255), -1)

    if not hand_found and prev_state is not None:
        hand_x, hand_y, hand_z = prev_state[0], prev_state[1], prev_state[2]

    state = np.array([hand_x, hand_y, hand_z,
                      cube_x, cube_y,
                      bowl_x, bowl_y], dtype=np.float32)
    return state, annotated, hand_found


def frame_to_tensor(frame_bgr, device: str):
    """Resize BGR frame to IMG_SIZE×IMG_SIZE and convert to [1,3,H,W] float32 (0-1)."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor  = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)


def run_inference(policy, img_tensor, state_vec, task: str, device: str):
    """Run a single SmolVLA forward pass and return the 4-dim action."""
    state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(device)
    obs = {
        "observation.images.phone": img_tensor,
        "observation.state":        state_tensor,
        "task":                     [task],
    }
    with torch.no_grad():
        action = policy.select_action(obs)
    return action.squeeze(0).cpu().numpy()          # shape (4,)


def decode_action(action: np.ndarray):
    """Human-readable action breakdown."""
    d_x, d_y, d_z = action[0], action[1], action[2]
    gripper_logit  = action[3]
    gripper_prob   = float(1.0 / (1.0 + np.exp(-gripper_logit)))
    gripper_str    = "CLOSED" if gripper_prob > 0.5 else "OPEN"
    return d_x, d_y, d_z, gripper_prob, gripper_str


def overlay_action(frame, action, state, fps=None):
    """Draw action + state overlay onto a BGR frame in place."""
    d_x, d_y, d_z, g_prob, g_str = decode_action(action)
    col_ok  = (0, 220, 80)
    col_bad = (0, 80, 220)
    g_col   = col_bad if g_prob > 0.5 else col_ok

    lines = [
        f"Action  dx={d_x:+.4f}  dy={d_y:+.4f}  dz={d_z:+.4f}",
        f"Gripper {g_str}  ({g_prob:.2f})",
        f"State   hand=({state[0]:.2f},{state[1]:.2f},{state[2]:.2f})",
        f"        cube=({state[3]:.2f},{state[4]:.2f})  bowl=({state[5]:.2f},{state[6]:.2f})",
    ]
    if fps is not None:
        lines.append(f"FPS     {fps:.1f}")

    y0 = 28
    for i, line in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    g_col if i == 1 else (255, 255, 255), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Source handlers
# ---------------------------------------------------------------------------

def run_on_image(args, policy, aruco_det, hand_det, device):
    """Run inference once on a static image and print results."""
    if not os.path.exists(args.path):
        print(f"❌  Image not found: {args.path}")
        sys.exit(1)

    frame = cv2.imread(args.path)
    print(f"\nRunning inference on: {args.path}")

    if args.detect:
        state, annotated, found = extract_state(frame, aruco_det, hand_det)
        print(f"   Hand detected: {found}")
    else:
        state    = np.array([0.5, 0.5, 0.0,
                             DEFAULT_CUBE[0], DEFAULT_CUBE[1],
                             DEFAULT_BOWL[0], DEFAULT_BOWL[1]], dtype=np.float32)
        annotated = frame.copy()

    img_tensor = frame_to_tensor(frame, device)
    t0 = time.time()
    action = run_inference(policy, img_tensor, state, TASK_DESCRIPTION, device)
    latency = (time.time() - t0) * 1000

    d_x, d_y, d_z, g_prob, g_str = decode_action(action)
    print(f"\n✅  Inference complete  ({latency:.1f} ms)")
    print(f"   Raw action vector  : {np.round(action, 4)}")
    print(f"   Translation delta  : dx={d_x:+.4f}  dy={d_y:+.4f}  dz={d_z:+.4f}")
    print(f"   Gripper            : {g_str}  (prob={g_prob:.4f})")
    print(f"\n   State used:")
    print(f"     hand  = ({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f})")
    print(f"     cube  = ({state[3]:.3f}, {state[4]:.3f})")
    print(f"     bowl  = ({state[5]:.3f}, {state[6]:.3f})")

    overlay_action(annotated, action, state)
    out_path = "scratch/inference_result.jpg"
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite(out_path, annotated)
    print(f"\n   Annotated frame saved → {out_path}")


def run_on_video(args, policy, aruco_det, hand_det, device):
    """Run frame-by-frame inference on a video file and save annotated output."""
    if not os.path.exists(args.path):
        print(f"❌  Video not found: {args.path}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.path)
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("scratch", exist_ok=True)
    out_path = "scratch/inference_annotated.mp4"
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps_in, (w, h))

    print(f"\nProcessing video: {args.path}  ({n_total} frames @ {fps_in:.1f} fps)")
    prev_state = None
    frame_idx  = 0
    t_start    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if args.detect:
            state, annotated, _ = extract_state(frame, aruco_det, hand_det, prev_state)
        else:
            state    = prev_state if prev_state is not None else np.array(
                [0.5, 0.5, 0.0,
                 DEFAULT_CUBE[0], DEFAULT_CUBE[1],
                 DEFAULT_BOWL[0], DEFAULT_BOWL[1]], dtype=np.float32)
            annotated = frame.copy()

        img_tensor = frame_to_tensor(frame, device)
        action     = run_inference(policy, img_tensor, state, TASK_DESCRIPTION, device)
        prev_state = state

        elapsed = time.time() - t_start
        fps_cur = frame_idx / elapsed if elapsed > 0 else 0
        overlay_action(annotated, action, state, fps=fps_cur)
        writer.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == n_total:
            d_x, d_y, d_z, g_prob, g_str = decode_action(action)
            print(f"  Frame {frame_idx:4d}/{n_total}  "
                  f"dx={d_x:+.4f} dy={d_y:+.4f} dz={d_z:+.4f}  "
                  f"grip={g_str} ({g_prob:.2f})  FPS={fps_cur:.1f}")

    cap.release()
    writer.release()
    print(f"\n✅  Annotated video saved → {out_path}")


def run_on_webcam(args, policy, aruco_det, hand_det, device):
    """Live webcam inference loop with real-time overlay (press Q to quit)."""
    cam_id = getattr(args, "cam_id", 0)
    cap    = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"❌  Could not open webcam (index {cam_id})")
        sys.exit(1)

    print(f"\n🎥  Live webcam inference  |  Press Q to quit")
    prev_state = None
    times      = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        if args.detect:
            state, annotated, _ = extract_state(frame, aruco_det, hand_det, prev_state)
        else:
            state    = prev_state if prev_state is not None else np.array(
                [0.5, 0.5, 0.0,
                 DEFAULT_CUBE[0], DEFAULT_CUBE[1],
                 DEFAULT_BOWL[0], DEFAULT_BOWL[1]], dtype=np.float32)
            annotated = frame.copy()

        img_tensor = frame_to_tensor(frame, device)
        action     = run_inference(policy, img_tensor, state, TASK_DESCRIPTION, device)
        prev_state = state

        elapsed = time.time() - t0
        times.append(elapsed)
        fps = 1.0 / (sum(times[-10:]) / len(times[-10:]))

        overlay_action(annotated, action, state, fps=fps)
        cv2.imshow("SmolVLA Inference  [Q to quit]", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅  Session ended.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SmolVLA inference — image / video / webcam",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--source",    choices=["image", "video", "webcam"],
                   default="webcam",
                   help="Input source  (default: webcam)")
    p.add_argument("--path",      default="scratch/sample_62.jpg",
                   help="Path to image or video file")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                   help="Path to pretrained_model directory")
    p.add_argument("--cam-id",    type=int, default=0,
                   help="Webcam device index  (default: 0)")
    p.add_argument("--no-detect", dest="detect", action="store_false",
                   default=True,
                   help="Skip hand/ArUco detection; use default state values")
    p.add_argument("--cpu",       action="store_true",
                   help="Force CPU even if CUDA is available")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # Load model
    policy = load_policy(args.checkpoint, device)

    # Setup detectors (only needed if --detect is active)
    if args.detect:
        if not os.path.exists("hand_landmarker.task"):
            print("⚠️  hand_landmarker.task not found — disabling detection (--no-detect mode).")
            args.detect = False
            aruco_det = hand_det = None
        else:
            aruco_det, hand_det = setup_detectors()
    else:
        aruco_det = hand_det = None

    print(f"\nTask  : \"{TASK_DESCRIPTION}\"")
    print(f"Source: {args.source.upper()}"
          + (f"  ({args.path})" if args.source != "webcam" else ""))
    print(f"Detect: {'hand + ArUco' if args.detect else 'disabled (default state)'}")
    print("-" * 60)

    if args.source == "image":
        run_on_image(args, policy, aruco_det, hand_det, device)
    elif args.source == "video":
        run_on_video(args, policy, aruco_det, hand_det, device)
    else:
        run_on_webcam(args, policy, aruco_det, hand_det, device)


if __name__ == "__main__":
    main()
