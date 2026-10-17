import cv2
import glob
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
# STEP 1: EXTRACT DEMONSTRATION TRAJECTORIES FROM EGOCENTRIC VIDEOS
# ----------------------------------------------------------------------
# Features extracted per timestep:
# - episode_id: demonstration index
# - step: frame index within episode
# - hand_x, hand_y, hand_z: 3D position of hand wrist
# - cube_x, cube_y: Pick Zone (ArUco ID 0)
# - bowl_x, bowl_y: Place Zone (ArUco ID 1)
# - gripper: 0.0 (Open) or 1.0 (Closed/Grasping)
# - Actions (Deltas): d_x, d_y, d_z, d_gripper
# ----------------------------------------------------------------------

def setup_detectors():
    # ArUco Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    return aruco_detector, hand_detector

def process_single_video(video_path, episode_id, aruco_detector, hand_detector):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    raw_frames_data = []
    
    # Track static zone locations as running average
    m0_positions = []
    m1_positions = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        h, w, _ = frame.shape

        # 1. Detect ArUco Markers
        corners, ids, _ = aruco_detector.detectMarkers(frame)
        c0, c1 = None, None
        if ids is not None:
            flat_ids = ids.flatten()
            for corner, m_id in zip(corners, flat_ids):
                pts = corner.reshape((4, 2))
                cx = float(np.mean(pts[:, 0])) / w
                cy = float(np.mean(pts[:, 1])) / h
                if m_id == 0:
                    c0 = (cx, cy)
                    m0_positions.append((cx, cy))
                elif m_id == 1:
                    c1 = (cx, cy)
                    m1_positions.append((cx, cy))

        # 2. Detect Hand
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_res = hand_detector.detect(mp_image)

        hand_found = False
        hx, hy, hz = 0.0, 0.0, 0.0
        pinch_dist = 1.0

        if hand_res.hand_landmarks and len(hand_res.hand_landmarks) > 0:
            hand_found = True
            lm = hand_res.hand_landmarks[0]
            # 0: Wrist, 4: Thumb Tip, 8: Index Tip
            wrist = lm[0]
            thumb = lm[4]
            index = lm[8]
            hx = float(wrist.x)
            hy = float(wrist.y)
            hz = float(wrist.z)
            pinch_dist = float(np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2 + (thumb.z - index.z)**2))

        raw_frames_data.append({
            "frame_idx": frame_idx,
            "hand_found": hand_found,
            "hx": hx, "hy": hy, "hz": hz,
            "pinch_dist": pinch_dist
        })

    cap.release()

    # QA Check: Filter videos with insufficient hand detection
    hand_frame_indices = [i for i, f in enumerate(raw_frames_data) if f["hand_found"]]
    if len(hand_frame_indices) < 35:
        print(f"Skipping {os.path.basename(video_path)}: only {len(hand_frame_indices)} active hand frames (QA threshold is 35)")
        return None

    # Reference zone coordinates (default to means or canonical bounds)
    cube_x = float(np.mean([p[0] for p in m0_positions])) if m0_positions else 0.75
    cube_y = float(np.mean([p[1] for p in m0_positions])) if m0_positions else 0.50
    bowl_x = float(np.mean([p[0] for p in m1_positions])) if m1_positions else 0.25
    bowl_y = float(np.mean([p[1] for p in m1_positions])) if m1_positions else 0.50

    # Window active demonstration from hand entry to hand retract
    start_idx = max(0, hand_frame_indices[0] - 2)
    end_idx = min(len(raw_frames_data), hand_frame_indices[-1] + 2)
    active_slice = raw_frames_data[start_idx:end_idx]

    # Interpolate missing hand frames within active window if any
    clean_trajectory = []
    last_valid_h = None
    for step_num, item in enumerate(active_slice):
        if item["hand_found"]:
            last_valid_h = (item["hx"], item["hy"], item["hz"])
            p_dist = item["pinch_dist"]
        elif last_valid_h is not None:
            # Carry forward last observed position
            item["hx"], item["hy"], item["hz"] = last_valid_h
            p_dist = 0.20
        else:
            continue

        # Gripper logic:
        # Manipulation progresses from Pick Zone (cube) to Place Zone (bowl).
        # Hand is grasped while transporting between Cube and Bowl.
        dist_to_cube = np.sqrt((item["hx"] - cube_x)**2 + (item["hy"] - cube_y)**2)
        dist_to_bowl = np.sqrt((item["hx"] - bowl_x)**2 + (item["hy"] - bowl_y)**2)

        # Closed gripper when hand has approached cube and is heading to / over bowl
        # Release occurs when hand drops into bowl or p_dist opens
        is_grasped = 1.0 if (item["hx"] < cube_x and item["hx"] > (bowl_x - 0.05)) else 0.0

        clean_trajectory.append({
            "episode_id": episode_id,
            "step": step_num,
            "hand_x": item["hx"],
            "hand_y": item["hy"],
            "hand_z": item["hz"],
            "cube_x": cube_x,
            "cube_y": cube_y,
            "bowl_x": bowl_x,
            "bowl_y": bowl_y,
            "gripper": is_grasped
        })

    if len(clean_trajectory) < 20:
        return None

    df_ep = pd.DataFrame(clean_trajectory)
    # Target state deltas: a_t = s_{t+1} - s_t
    df_ep['d_x'] = df_ep['hand_x'].shift(-1) - df_ep['hand_x']
    df_ep['d_y'] = df_ep['hand_y'].shift(-1) - df_ep['hand_y']
    df_ep['d_z'] = df_ep['hand_z'].shift(-1) - df_ep['hand_z']
    df_ep['d_gripper'] = df_ep['gripper'].shift(-1)

    # Drop last step (no target delta)
    df_ep = df_ep.dropna()
    return df_ep

def build_dataset(videos_dir="raw_videos", output_csv="dataset.csv"):
    video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    print(f"Discovered {len(video_files)} video files in '{videos_dir}'.")
    if not video_files:
        print("No videos found. Aborting.")
        return

    aruco_det, hand_det = setup_detectors()
    all_episodes = []

    valid_ep_count = 0
    for v_path in video_files:
        print(f"Processing Demonstration #{valid_ep_count + 1}: {os.path.basename(v_path)}...")
        df_ep = process_single_video(v_path, valid_ep_count, aruco_det, hand_det)
        if df_ep is not None and len(df_ep) > 0:
            all_episodes.append(df_ep)
            valid_ep_count += 1
            print(f"  -> Added {len(df_ep)} valid timesteps.")

    if not all_episodes:
        print("No valid demonstrations could be processed.")
        return

    full_dataset = pd.concat(all_episodes, ignore_index=True)
    full_dataset.to_csv(output_csv, index=False)

    print("\n================ DATASET CREATION SUMMARY ================")
    print(f"Successfully Processed Demonstrations: {valid_ep_count}/{len(video_files)}")
    print(f"Total State-Action Pairs: {len(full_dataset)}")
    print(f"Saved Dataset: {output_csv}")
    print(f"Columns: {list(full_dataset.columns)}")
    print("Sample Data:")
    print(full_dataset.head(3))
    print("==========================================================")

if __name__ == "__main__":
    build_dataset()
