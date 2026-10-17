"""
agents/annotation_agent.py
==========================
Annotation Agent — per-frame annotation of hand landmarks, ArUco markers,
grasp events, and task phase labels. Generates structured annotations that
can be used for dataset visualisation or future model training.

Outputs:
  - annotations/<episode_id>.json  per-episode frame annotations
  - scratch/annotated_<video>.mp4  visualisation videos

Usage
-----
  from agents.annotation_agent import AnnotationAgent
  agent = AnnotationAgent()
  agent.run()
"""

import cv2
import glob
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime


class AnnotationAgent:
    """
    Automated annotation agent for egocentric demonstration videos.

    Extracts per-frame:
      - Hand wrist position (normalised x, y, z)
      - Thumb-index pinch distance (gripper proxy)
      - ArUco marker positions for cube (ID 0) and bowl (ID 1)
      - Estimated task phase: approach / grasp / transport / release / retract
    """

    TASK_PHASES = {
        "approach":  "Moving toward cube",
        "grasp":     "Closing gripper on cube",
        "transport": "Carrying cube to bowl",
        "release":   "Releasing cube into bowl",
        "retract":   "Hand returning to neutral",
        "idle":      "No active manipulation",
    }

    def __init__(
        self,
        videos_dir="raw_videos",
        output_dir="annotations",
        hand_model="hand_landmarker.task",
        save_videos=False,
        pinch_close_threshold=0.06,
    ):
        self.videos_dir           = videos_dir
        self.output_dir           = output_dir
        self.hand_model           = hand_model
        self.save_videos          = save_videos
        self.pinch_close_threshold = pinch_close_threshold
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _setup_detectors(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_det  = cv2.aruco.ArucoDetector(aruco_dict,
                                              cv2.aruco.DetectorParameters())
        base_opts  = mp_python.BaseOptions(model_asset_path=self.hand_model)
        options    = mp_vision.HandLandmarkerOptions(
            base_options=base_opts, num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        hand_det = mp_vision.HandLandmarker.create_from_options(options)
        return aruco_det, hand_det

    # ------------------------------------------------------------------
    def _classify_phase(self, hand_x, cube_x, bowl_x, gripper_closed,
                         cube_reached, cube_placed, hand_found):
        """Simple finite-state task phase classifier."""
        if not hand_found:
            return "idle"
        if cube_placed:
            return "retract"
        if gripper_closed and cube_reached:
            if hand_x < cube_x:          # moving toward bowl
                return "transport"
            return "grasp"
        if not gripper_closed and not cube_reached:
            return "approach"
        if not gripper_closed and cube_reached:
            return "release"
        return "approach"

    # ------------------------------------------------------------------
    def _annotate_video(self, video_path, episode_id, aruco_det, hand_det):
        """Annotate one video and return list of per-frame dicts."""
        import mediapipe as mp

        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.save_videos:
            os.makedirs("scratch", exist_ok=True)
            out_path = f"scratch/annotated_{os.path.basename(video_path)}"
            writer   = cv2.VideoWriter(out_path,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (W, H))

        frames          = []
        cube_positions  = []
        bowl_positions  = []
        cube_placed     = False
        frame_idx       = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # --- ArUco ---
            corners, ids, _ = aruco_det.detectMarkers(frame)
            cube_xy = bowl_xy = None
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                for corner, mid in zip(corners, ids.flatten()):
                    pts = corner.reshape((4, 2))
                    cx  = float(np.mean(pts[:, 0])) / W
                    cy  = float(np.mean(pts[:, 1])) / H
                    if mid == 0:
                        cube_xy = (cx, cy)
                        cube_positions.append((cx, cy))
                    elif mid == 1:
                        bowl_xy = (cx, cy)
                        bowl_positions.append((cx, cy))

            # --- Hand ---
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_res = hand_det.detect(mp_img)

            hand_found = False
            hx = hy = hz = 0.0
            pinch_dist   = 1.0
            gripper_closed = False

            if hand_res.hand_landmarks:
                hand_found = True
                lm     = hand_res.hand_landmarks[0]
                wrist  = lm[0]; thumb = lm[4]; index = lm[8]
                hx, hy, hz = float(wrist.x), float(wrist.y), float(wrist.z)
                pinch_dist = float(np.sqrt(
                    (thumb.x-index.x)**2 + (thumb.y-index.y)**2 + (thumb.z-index.z)**2
                ))
                gripper_closed = pinch_dist < self.pinch_close_threshold

                for lmk in lm:
                    cv2.circle(frame, (int(lmk.x*W), int(lmk.y*H)), 3,
                               (0,255,0), -1)
                cv2.circle(frame, (int(hx*W), int(hy*H)), 8,
                           (0,0,255) if gripper_closed else (0,200,0), -1)

            # Use running mean for missing markers
            cube_x_ref = (float(np.mean([p[0] for p in cube_positions]))
                          if cube_positions else 0.75)
            bowl_x_ref = (float(np.mean([p[0] for p in bowl_positions]))
                          if bowl_positions else 0.25)

            cube_reached = cube_xy is not None and hand_found and (
                abs(hx - cube_xy[0]) < 0.15 and abs(hy - cube_xy[1]) < 0.15
            )
            if (bowl_xy is not None and hand_found and
                    not gripper_closed and cube_reached):
                cube_placed = True

            phase = self._classify_phase(
                hx, cube_x_ref, bowl_x_ref,
                gripper_closed, cube_reached, cube_placed, hand_found
            )

            # HUD on frame
            cv2.putText(frame, f"Phase: {phase}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,100), 2)
            cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.putText(frame, f"Grip: {'CLOSED' if gripper_closed else 'OPEN'}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0,0,255) if gripper_closed else (0,200,0), 1)

            if writer:
                writer.write(frame)

            frames.append({
                "frame":           frame_idx,
                "timestamp_s":     round(frame_idx / fps, 3),
                "hand_found":      hand_found,
                "hand_x":          round(hx, 4),
                "hand_y":          round(hy, 4),
                "hand_z":          round(hz, 4),
                "pinch_dist":      round(pinch_dist, 4),
                "gripper_closed":  gripper_closed,
                "cube_visible":    cube_xy is not None,
                "cube_x":          round(cube_xy[0], 4) if cube_xy else None,
                "cube_y":          round(cube_xy[1], 4) if cube_xy else None,
                "bowl_visible":    bowl_xy is not None,
                "bowl_x":          round(bowl_xy[0], 4) if bowl_xy else None,
                "bowl_y":          round(bowl_xy[1], 4) if bowl_xy else None,
                "task_phase":      phase,
                "cube_reached":    cube_reached,
                "cube_placed":     cube_placed,
            })

        cap.release()
        if writer:
            writer.release()

        return frames

    # ------------------------------------------------------------------
    def run(self, verbose=True):
        """Annotate all videos and save per-episode JSON files."""
        if not os.path.exists(self.hand_model):
            print(f"[AnnotationAgent] ❌  hand_landmarker.task not found at '{self.hand_model}'")
            return

        videos = sorted(glob.glob(os.path.join(self.videos_dir, "*.mp4")))
        if not videos:
            print(f"[AnnotationAgent] No videos found in '{self.videos_dir}'")
            return

        aruco_det, hand_det = self._setup_detectors()

        summary = {"generated_at": datetime.utcnow().isoformat()+"Z",
                   "episodes": []}

        for ep_id, vpath in enumerate(videos):
            vname = os.path.basename(vpath)
            if verbose:
                print(f"[AnnotationAgent] Ep {ep_id:02d}: {vname}")

            frames = self._annotate_video(vpath, ep_id, aruco_det, hand_det)

            out_path = os.path.join(self.output_dir, f"episode_{ep_id:04d}.json")
            with open(out_path, "w") as f:
                json.dump({
                    "episode_id": ep_id,
                    "video":      vname,
                    "n_frames":   len(frames),
                    "frames":     frames,
                }, f, indent=2)

            phases = [f["task_phase"] for f in frames]
            summary["episodes"].append({
                "episode_id": ep_id,
                "video":      vname,
                "n_frames":   len(frames),
                "phases":     {p: phases.count(p) for p in set(phases)},
                "output":     out_path,
            })
            if verbose:
                hand_ct = sum(1 for f in frames if f["hand_found"])
                print(f"  → {len(frames)} frames | hand={hand_ct} | "
                      f"phases={set(phases)}")

        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[AnnotationAgent] Done. Annotations → {self.output_dir}/")


if __name__ == "__main__":
    agent = AnnotationAgent(save_videos=True)
    agent.run()
