import cv2
import os
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def inspect_video():
    videos = glob.glob("raw_videos/*.mp4")
    if not videos:
        print("No videos found in raw_videos!")
        return

    video_path = videos[0]
    print(f"Testing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Properties: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({total_frames/fps:.2f}s)")

    # 1. Setup ArUco Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 2. Setup MediaPipe HandLandmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    m0_detected = 0
    m1_detected = 0
    hand_detected = 0

    hand_trajectory = []
    pinch_distances = []
    frames_processed = 0

    os.makedirs("scratch", exist_ok=True)
    diagnostic_frames = [
        int(total_frames * 0.1),
        int(total_frames * 0.3),
        int(total_frames * 0.5),
        int(total_frames * 0.7),
        int(total_frames * 0.9)
    ]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_processed += 1
        h, w, _ = frame.shape

        # A. ArUco Detection
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            flat_ids = ids.flatten()
            if 0 in flat_ids:
                m0_detected += 1
            if 1 in flat_ids:
                m1_detected += 1

        # B. Hand Detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_res = hand_detector.detect(mp_image)

        wrist_pt = None
        if hand_res.hand_landmarks and len(hand_res.hand_landmarks) > 0:
            hand_detected += 1
            landmarks = hand_res.hand_landmarks[0]
            # 0: Wrist, 4: Thumb Tip, 8: Index Tip
            wrist = landmarks[0]
            thumb = landmarks[4]
            index = landmarks[8]

            wrist_pt = (int(wrist.x * w), int(wrist.y * h))
            hand_trajectory.append(wrist_pt)

            pinch_dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2 + (thumb.z - index.z)**2)
            pinch_distances.append(pinch_dist)

        # C. Save Diagnostic Snapshots
        if frames_processed in diagnostic_frames:
            annotated = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            if hand_res.hand_landmarks and len(hand_res.hand_landmarks) > 0:
                landmarks = hand_res.hand_landmarks[0]
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
                if wrist_pt:
                    cv2.putText(annotated, f"Pinch: {pinch_distances[-1]:.3f}", (wrist_pt[0], wrist_pt[1]-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(f"scratch/sample_{frames_processed}.jpg", annotated)

    cap.release()

    print("\n================== EVALUATION REPORT ==================")
    print(f"Total Frames: {frames_processed}")
    print(f"Marker 0 (Pick Zone / Cube) Detection Rate: {m0_detected}/{frames_processed} ({m0_detected/frames_processed*100:.1f}%)")
    print(f"Marker 1 (Place Zone / Bowl) Detection Rate: {m1_detected}/{frames_processed} ({m1_detected/frames_processed*100:.1f}%)")
    print(f"Hand Detection Rate: {hand_detected}/{frames_processed} ({hand_detected/frames_processed*100:.1f}%)")

    if pinch_distances:
        p_min = min(pinch_distances)
        p_max = max(pinch_distances)
        p_mean = np.mean(pinch_distances)
        print(f"Pinch Distances: Min={p_min:.4f}, Max={p_max:.4f}, Mean={p_mean:.4f}")
        # Count pinched frames
        threshold = (p_min + p_max) / 2
        grasped_frames = sum(1 for p in pinch_distances if p < threshold)
        print(f"Dynamic Grasp Threshold: {threshold:.4f} -> Grasped Frames: {grasped_frames}/{len(pinch_distances)} ({grasped_frames/len(pinch_distances)*100:.1f}%)")

    if hand_trajectory:
        xs = [p[0] for p in hand_trajectory]
        ys = [p[1] for p in hand_trajectory]
        print(f"Hand Pixel Workspace: X: [{min(xs)}, {max(xs)}] (Span: {max(xs)-min(xs)}px), Y: [{min(ys)}, {max(ys)}] (Span: {max(ys)-min(ys)}px)")
    print("=======================================================")

if __name__ == "__main__":
    inspect_video()
