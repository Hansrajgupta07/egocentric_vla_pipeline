import cv2
import glob
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def batch_validate():
    videos = sorted(glob.glob("raw_videos/*.mp4"))
    print(f"Total videos found: {len(videos)}")
    if not videos:
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    stats = []

    for idx, vpath in enumerate(videos, 1):
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        m0_count = 0
        m1_count = 0
        hand_frames = []
        pinch_vals = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ArUco
            corners, ids, _ = detector.detectMarkers(frame)
            if ids is not None:
                flat = ids.flatten()
                if 0 in flat:
                    m0_count += 1
                if 1 in flat:
                    m1_count += 1

            # Hand
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                hand_frames.append(frame_idx)
                lm = res.hand_landmarks[0]
                thumb = lm[4]
                index = lm[8]
                dist = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2 + (thumb.z - index.z)**2)
                pinch_vals.append(dist)

        cap.release()

        hand_span = (max(hand_frames) - min(hand_frames) + 1) if hand_frames else 0
        hand_detected = len(hand_frames)
        m0_pct = (m0_count / total_frames * 100) if total_frames else 0
        m1_pct = (m1_count / total_frames * 100) if total_frames else 0
        vname = os.path.basename(vpath)

        stats.append({
            "idx": idx,
            "filename": vname,
            "frames": total_frames,
            "duration": duration,
            "m0_pct": m0_pct,
            "m1_pct": m1_pct,
            "hand_frames": hand_detected,
            "hand_span": hand_span,
            "start_frame": min(hand_frames) if hand_frames else 0,
            "end_frame": max(hand_frames) if hand_frames else 0,
            "p_min": min(pinch_vals) if pinch_vals else 0,
            "p_max": max(pinch_vals) if pinch_vals else 0
        })

        print(f"[{idx:02d}/25] {vname[:32]}... | Frames: {total_frames} ({duration:.1f}s) | M0: {m0_pct:.0f}% | M1: {m1_pct:.0f}% | Hand: {hand_detected} f (Span: {hand_span} f)")

    print("\n================ DATASET VALIDATION SUMMARY ================")
    all_frames = sum(s['frames'] for s in stats)
    all_duration = sum(s['duration'] for s in stats)
    avg_m0 = np.mean([s['m0_pct'] for s in stats])
    avg_m1 = np.mean([s['m1_pct'] for s in stats])
    total_hand_frames = sum(s['hand_frames'] for s in stats)

    print(f"Total Demonstrations: {len(stats)}")
    print(f"Total Video Time: {all_duration:.1f} seconds (~{all_duration/60:.1f} minutes)")
    print(f"Total Video Frames: {all_frames} frames")
    print(f"Total Active Hand Manipulation Frames: {total_hand_frames} frames")
    print(f"Average Marker 0 Tracking: {avg_m0:.1f}%")
    print(f"Average Marker 1 Tracking: {avg_m1:.1f}%")
    print("============================================================")

if __name__ == "__main__":
    batch_validate()
