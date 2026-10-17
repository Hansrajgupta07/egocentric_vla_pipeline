"""
agents/dataset_agent.py
=======================
Dataset Agent — QA, validation, and metadata generation for raw demonstration videos.

Responsibilities:
  1. Scan raw_videos/ and count demonstrations
  2. Check video properties (resolution, FPS, duration)
  3. Verify ArUco marker detectability per video
  4. Verify hand detectability per video
  5. Flag poor-quality demonstrations
  6. Generate train/validation episode splits
  7. Save dataset_meta.json with full audit trail

Usage
-----
  from agents.dataset_agent import DatasetAgent
  agent = DatasetAgent()
  report = agent.run()
"""

import cv2
import glob
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path


class DatasetAgent:
    """
    Automated QA agent for the raw egocentric demonstration dataset.

    Parameters
    ----------
    videos_dir    : str  Path to raw_videos/
    output_json   : str  Where to save the audit report
    hand_model    : str  Path to hand_landmarker.task
    min_hand_frames : int  Minimum detected hand frames to pass QA
    val_fraction  : float  Fraction of episodes held out for validation
    """

    def __init__(
        self,
        videos_dir="raw_videos",
        output_json="dataset_meta.json",
        hand_model="hand_landmarker.task",
        min_hand_frames=35,
        val_fraction=0.2,
        seed=42,
    ):
        self.videos_dir      = videos_dir
        self.output_json     = output_json
        self.hand_model      = hand_model
        self.min_hand_frames = min_hand_frames
        self.val_fraction    = val_fraction
        self.seed            = seed

    # ------------------------------------------------------------------
    def _setup_detectors(self):
        """Initialise ArUco + MediaPipe detectors."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_det   = cv2.aruco.ArucoDetector(aruco_dict,
                                               cv2.aruco.DetectorParameters())

        base_opts   = mp_python.BaseOptions(model_asset_path=self.hand_model)
        options     = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        hand_det    = mp_vision.HandLandmarker.create_from_options(options)
        return aruco_det, hand_det

    # ------------------------------------------------------------------
    def _inspect_video(self, video_path, aruco_det, hand_det):
        """Return a metadata dict for one video."""
        import mediapipe as mp

        cap         = cv2.VideoCapture(video_path)
        fps         = cap.get(cv2.CAP_PROP_FPS) or 30
        n_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s  = n_frames / fps

        m0_count = m1_count = hand_count = 0
        pinch_vals = []
        hand_frames_idx = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            corners, ids, _ = aruco_det.detectMarkers(frame)
            if ids is not None:
                flat = ids.flatten()
                if 0 in flat: m0_count += 1
                if 1 in flat: m1_count += 1

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res     = hand_det.detect(mp_img)
            if res.hand_landmarks:
                hand_count += 1
                hand_frames_idx.append(frame_idx)
                lm    = res.hand_landmarks[0]
                thumb = lm[4]; index = lm[8]
                d     = float(np.sqrt((thumb.x-index.x)**2 +
                                      (thumb.y-index.y)**2 +
                                      (thumb.z-index.z)**2))
                pinch_vals.append(d)

        cap.release()

        passes_qa = hand_count >= self.min_hand_frames
        return {
            "filename":     os.path.basename(video_path),
            "resolution":   f"{width}x{height}",
            "fps":          round(fps, 1),
            "n_frames":     n_frames,
            "duration_s":   round(duration_s, 2),
            "m0_detected":  m0_count,
            "m1_detected":  m1_count,
            "m0_pct":       round(m0_count / max(n_frames, 1) * 100, 1),
            "m1_pct":       round(m1_count / max(n_frames, 1) * 100, 1),
            "hand_frames":  hand_count,
            "hand_pct":     round(hand_count / max(n_frames, 1) * 100, 1),
            "pinch_min":    round(float(min(pinch_vals)), 4) if pinch_vals else None,
            "pinch_max":    round(float(max(pinch_vals)), 4) if pinch_vals else None,
            "passes_qa":    passes_qa,
            "qa_reason":    "" if passes_qa else f"hand_frames={hand_count} < {self.min_hand_frames}",
        }

    # ------------------------------------------------------------------
    def _make_splits(self, valid_videos):
        """Create train/val episode splits deterministically."""
        rng = np.random.default_rng(self.seed)
        idxs = np.arange(len(valid_videos))
        rng.shuffle(idxs)
        n_val  = max(1, int(len(idxs) * self.val_fraction))
        val_idxs   = sorted(idxs[:n_val].tolist())
        train_idxs = sorted(idxs[n_val:].tolist())
        return {
            "train": [valid_videos[i] for i in train_idxs],
            "val":   [valid_videos[i] for i in val_idxs],
        }

    # ------------------------------------------------------------------
    def run(self, verbose=True):
        """
        Run the full dataset QA pipeline.
        Returns the report dict and saves it to output_json.
        """
        videos = sorted(glob.glob(os.path.join(self.videos_dir, "*.mp4")))
        if not videos:
            print(f"[DatasetAgent] ⚠️  No .mp4 files found in '{self.videos_dir}'")
            return {}

        if verbose:
            print(f"\n[DatasetAgent] Scanning {len(videos)} videos in '{self.videos_dir}' ...")

        has_hand_model = os.path.exists(self.hand_model)
        if has_hand_model:
            aruco_det, hand_det = self._setup_detectors()
        else:
            print(f"[DatasetAgent] ⚠️  hand_landmarker.task not found — skipping detection QA.")
            aruco_det = hand_det = None

        video_reports = []
        for i, vpath in enumerate(videos, 1):
            if verbose:
                print(f"  [{i:02d}/{len(videos)}] {os.path.basename(vpath)}", end="", flush=True)
            if has_hand_model:
                meta = self._inspect_video(vpath, aruco_det, hand_det)
            else:
                cap      = cv2.VideoCapture(vpath)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps      = cap.get(cv2.CAP_PROP_FPS) or 30
                w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                meta = {
                    "filename": os.path.basename(vpath),
                    "resolution": f"{w}x{h}",
                    "fps": round(fps, 1),
                    "n_frames": n_frames,
                    "duration_s": round(n_frames / fps, 2),
                    "passes_qa": True,
                    "qa_reason": "detection skipped (no hand_landmarker.task)",
                }
            video_reports.append(meta)
            if verbose:
                status = "✅" if meta.get("passes_qa", True) else f"❌  {meta.get('qa_reason', '')}"
                print(f"  → {status}")

        valid   = [m["filename"] for m in video_reports if m.get("passes_qa", True)]
        invalid = [m["filename"] for m in video_reports if not m.get("passes_qa", True)]
        splits  = self._make_splits(valid)

        report = {
            "generated_at":    datetime.utcnow().isoformat() + "Z",
            "videos_dir":      self.videos_dir,
            "total_videos":    len(videos),
            "valid_videos":    len(valid),
            "invalid_videos":  len(invalid),
            "train_episodes":  len(splits["train"]),
            "val_episodes":    len(splits["val"]),
            "qa_threshold":    self.min_hand_frames,
            "val_fraction":    self.val_fraction,
            "splits":          splits,
            "rejected":        invalid,
            "per_video":       video_reports,
        }

        with open(self.output_json, "w") as f:
            json.dump(report, f, indent=2)

        if verbose:
            print(f"\n[DatasetAgent] Summary")
            print(f"  Total videos  : {report['total_videos']}")
            print(f"  Valid (pass QA): {report['valid_videos']}")
            print(f"  Rejected       : {report['invalid_videos']}")
            print(f"  Train split    : {report['train_episodes']} episodes")
            print(f"  Val split      : {report['val_episodes']} episodes")
            print(f"  Report saved   → {self.output_json}")

        return report


if __name__ == "__main__":
    agent = DatasetAgent()
    agent.run()
