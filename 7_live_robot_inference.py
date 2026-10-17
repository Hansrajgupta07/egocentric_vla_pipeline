"""
7_live_robot_inference.py
=========================
Real-time inference using the fine-tuned SmolVLA checkpoint for
LIVE hardware deployment (phone/webcam → robot controller).

Pipeline per frame:
  1. Capture frame from phone/webcam
  2. Detect hand (wrist) position with MediaPipe
  3. Detect zone markers (ArUco ID 0 = cube, ID 1 = bowl)
  4. Build 7-dim state vector
  5. Run SmolVLA inference → 4-dim action [dx, dy, dz, gripper]
  6. Send action to robot controller (serial / ROS / mock)

Robot Controller Modes
----------------------
  --robot mock    : Print actions to console only (no hardware required)
  --robot serial  : Send over serial port (set --port /dev/ttyUSB0)
  --robot ros     : Publish to /robot_action ROS topic (requires rospy)

Usage
-----
  python 7_live_robot_inference.py --robot mock
  python 7_live_robot_inference.py --robot serial --port /dev/ttyUSB0 --baud 115200
  python 7_live_robot_inference.py --robot mock --cam-id 1 --fps-limit 10
"""

import argparse
import time
import os
import sys
import struct
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Config (must match training)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "smolvla_checkpoint-20261010T074618Z-1-001"
    "/smolvla_checkpoint/checkpoints/010000/pretrained_model"
)
TASK_DESCRIPTION = "Pick up the Rubik's cube and place it inside the bowl"
IMG_SIZE         = 224
STATE_DIM        = 7     # [hand_x, hand_y, hand_z, cube_x, cube_y, bowl_x, bowl_y]
ACTION_DIM       = 4     # [d_x, d_y, d_z, d_gripper_logit]
DEFAULT_CUBE     = (0.75, 0.50)
DEFAULT_BOWL     = (0.25, 0.50)

# ---------------------------------------------------------------------------
# Robot controller interface
# ---------------------------------------------------------------------------

class MockRobot:
    """Prints actions to console — no hardware required."""
    def __init__(self, **_):
        print("🤖  MockRobot: actions will be printed to console only.")

    def send(self, action: np.ndarray):
        d_x, d_y, d_z = action[0], action[1], action[2]
        g   = float(1 / (1 + np.exp(-action[3])))
        gst = "CLOSE" if g > 0.5 else "OPEN "
        print(f"   ACTION  dx={d_x:+.4f}  dy={d_y:+.4f}  dz={d_z:+.4f}  "
              f"gripper={gst} ({g:.2f})")

    def close(self): pass


class SerialRobot:
    """
    Sends 4 float32 values over a serial connection.
    Protocol: [0xAA] [dx f32] [dy f32] [dz f32] [gripper_prob f32] [0xBB]
    """
    def __init__(self, port="/dev/ttyUSB0", baud=115200, **_):
        import serial
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)   # allow Arduino/controller to reset
        print(f"🤖  SerialRobot: connected to {port} @ {baud} baud")

    def send(self, action: np.ndarray):
        g = float(1 / (1 + np.exp(-action[3])))
        payload = struct.pack("<4f", float(action[0]), float(action[1]),
                              float(action[2]), g)
        self.ser.write(b"\xAA" + payload + b"\xBB")

    def close(self):
        self.ser.close()


class ROSRobot:
    """Publishes Float32MultiArray to /robot_action (requires rospy)."""
    def __init__(self, **_):
        import rospy
        from std_msgs.msg import Float32MultiArray
        rospy.init_node("smolvla_inference", anonymous=True)
        self.pub = rospy.Publisher("/robot_action", Float32MultiArray, queue_size=1)
        self.msg_cls = Float32MultiArray
        print("🤖  ROSRobot: publishing to /robot_action")

    def send(self, action: np.ndarray):
        import rospy
        g = float(1 / (1 + np.exp(-action[3])))
        msg = self.msg_cls()
        msg.data = [float(action[0]), float(action[1]), float(action[2]), g]
        self.pub.publish(msg)

    def close(self): pass


ROBOT_MAP = {"mock": MockRobot, "serial": SerialRobot, "ros": ROSRobot}

# ---------------------------------------------------------------------------
# Perception helpers
# ---------------------------------------------------------------------------

def setup_detectors():
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


def extract_state(frame_bgr, aruco_det, hand_det, prev_state=None):
    h, w    = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()

    corners, ids, _ = aruco_det.detectMarkers(frame_bgr)
    cube_x, cube_y  = DEFAULT_CUBE
    bowl_x, bowl_y  = DEFAULT_BOWL

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for corner, mid in zip(corners, ids.flatten()):
            pts = corner.reshape((4, 2))
            cx  = float(np.mean(pts[:, 0])) / w
            cy  = float(np.mean(pts[:, 1])) / h
            if mid == 0: cube_x, cube_y = cx, cy
            elif mid == 1: bowl_x, bowl_y = cx, cy

    rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_res = hand_det.detect(mp_img)

    hand_x = hand_y = hand_z = 0.5
    if hand_res.hand_landmarks:
        lm     = hand_res.hand_landmarks[0]
        wrist  = lm[0]
        hand_x = float(wrist.x)
        hand_y = float(wrist.y)
        hand_z = float(wrist.z)
        for lmk in lm:
            cv2.circle(annotated, (int(lmk.x * w), int(lmk.y * h)), 4, (0, 255, 0), -1)
        cv2.circle(annotated, (int(hand_x * w), int(hand_y * h)), 8, (0, 0, 255), -1)
    elif prev_state is not None:
        hand_x, hand_y, hand_z = prev_state[0], prev_state[1], prev_state[2]

    state = np.array([hand_x, hand_y, hand_z,
                      cube_x, cube_y,
                      bowl_x, bowl_y], dtype=np.float32)
    return state, annotated


def frame_to_tensor(frame_bgr, device):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor  = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SmolVLA live robot inference")
    parser.add_argument("--robot",      choices=["mock", "serial", "ros"],
                        default="mock")
    parser.add_argument("--port",       default="/dev/ttyUSB0")
    parser.add_argument("--baud",       type=int, default=115200)
    parser.add_argument("--cam-id",     type=int, default=0)
    parser.add_argument("--fps-limit",  type=float, default=10.0,
                        help="Max inference Hz (default: 10)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--cpu",        action="store_true")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV preview window (headless mode)")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # --- Load SmolVLA ---
    if not os.path.isdir(args.checkpoint):
        print(f"❌  Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print(f"Loading SmolVLA from {args.checkpoint} ...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    print(f"✅  Model ready on {device.upper()}")

    # --- Setup detectors ---
    if not os.path.exists("hand_landmarker.task"):
        print("⚠️  hand_landmarker.task not found — state will use defaults.")
        aruco_det = hand_det = None
    else:
        aruco_det, hand_det = setup_detectors()

    # --- Open camera ---
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"❌  Could not open webcam {args.cam_id}")
        sys.exit(1)

    # --- Robot controller ---
    robot = ROBOT_MAP[args.robot](port=args.port, baud=args.baud)

    print(f"\n🚀  Live inference loop started")
    print(f"   Task  : {TASK_DESCRIPTION}")
    print(f"   Robot : {args.robot.upper()}")
    print(f"   Device: {device.upper()}")
    print(f"   FPS   : ≤ {args.fps_limit}")
    print("   Press Q in the preview window (or Ctrl+C) to quit.\n")

    period     = 1.0 / args.fps_limit
    prev_state = None
    step       = 0
    t_session  = time.time()

    try:
        while True:
            t_iter = time.time()

            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame capture failed — retrying...")
                time.sleep(0.1)
                continue

            # Perception
            if aruco_det is not None and hand_det is not None:
                state, annotated = extract_state(frame, aruco_det, hand_det, prev_state)
            else:
                state = prev_state if prev_state is not None else np.array(
                    [0.5, 0.5, 0.0,
                     DEFAULT_CUBE[0], DEFAULT_CUBE[1],
                     DEFAULT_BOWL[0], DEFAULT_BOWL[1]], dtype=np.float32)
                annotated = frame.copy()
            prev_state = state

            # SmolVLA inference
            img_tensor   = frame_to_tensor(frame, device)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            obs = {
                "observation.images.phone": img_tensor,
                "observation.state":        state_tensor,
                "task":                     [TASK_DESCRIPTION],
            }
            with torch.no_grad():
                action = policy.select_action(obs).squeeze(0).cpu().numpy()

            # Send to robot
            robot.send(action)
            step += 1

            # HUD overlay
            g = float(1 / (1 + np.exp(-action[3])))
            g_str = "CLOSED" if g > 0.5 else "OPEN"
            elapsed  = time.time() - t_session
            live_fps = step / elapsed if elapsed > 0 else 0
            hud_lines = [
                f"Step {step:04d}  FPS={live_fps:.1f}",
                f"dx={action[0]:+.4f}  dy={action[1]:+.4f}  dz={action[2]:+.4f}",
                f"Gripper: {g_str} ({g:.2f})",
                f"Hand: ({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f})",
            ]
            for i, line in enumerate(hud_lines):
                col = (0, 80, 220) if (i == 2 and g > 0.5) else (255, 255, 255)
                cv2.putText(annotated, line, (10, 26 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

            if not args.no_display:
                cv2.imshow("SmolVLA Live  [Q=quit]", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Rate limiting
            sleep_t = period - (time.time() - t_iter)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n🛑  Stopped by user.")

    finally:
        cap.release()
        robot.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        total = time.time() - t_session
        print(f"\n📊  Session: {step} steps in {total:.1f}s  "
              f"(avg {step/total:.1f} Hz)")


if __name__ == "__main__":
    main()
