import cv2, os, json, glob
import numpy as np
import pandas as pd
from pathlib import Path

RAW_VIDEOS_DIR   = "raw_videos"
DATASET_CSV      = "dataset.csv"
OUTPUT_DIR       = "lerobot_dataset"
TASK_DESCRIPTION = "Pick up the Rubik's cube and place it inside the bowl"
FPS              = 24
IMG_W, IMG_H     = 224, 224
CAMERA_KEY       = "observation.images.phone"
ACTION_COLS      = ["d_x", "d_y", "d_z", "d_gripper"]
STATE_COLS       = ["hand_x", "hand_y", "hand_z", "cube_x", "cube_y", "bowl_x", "bowl_y"]


def extract_frames(video_path, start_frame, n_frames):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()
    return frames


def write_video(frames, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def compute_stats(df):
    stats = {}
    for col in ACTION_COLS + STATE_COLS:
        if col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std":  float(df[col].std()) if df[col].std() > 0 else 1.0,
                "min":  float(df[col].min()),
                "max":  float(df[col].max()),
            }
    stats[CAMERA_KEY] = {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "min":  [0.0, 0.0, 0.0],
        "max":  [1.0, 1.0, 1.0],
    }
    return stats


def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df          = pd.read_csv(DATASET_CSV)
    video_files = sorted(glob.glob(os.path.join(RAW_VIDEOS_DIR, "*.mp4")))
    episodes    = sorted(df["episode_id"].unique())

    print(f"Loaded {len(df)} rows | {len(episodes)} episodes | {len(video_files)} videos")

    data_dir  = Path(OUTPUT_DIR) / "data"   / "chunk-000"
    video_dir = Path(OUTPUT_DIR) / "videos" / CAMERA_KEY / "chunk-000"
    meta_dir  = Path(OUTPUT_DIR) / "meta"
    for d in [data_dir, video_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    episodes_meta     = []
    stats_actions_dfs = []
    stats_states_dfs  = []
    global_frame_idx  = 0

    for ep_idx, ep_id in enumerate(episodes):
        ep_df    = df[df["episode_id"] == ep_id].reset_index(drop=True)
        n_frames = len(ep_df)
        if n_frames < 10 or ep_idx >= len(video_files):
            print(f"  Skip ep {ep_id}")
            continue

        vpath      = video_files[ep_idx]
        start_step = int(ep_df["step"].iloc[0])
        print(f"  Ep {ep_id:02d}: {n_frames} frames <- {os.path.basename(vpath)}")

        frames = extract_frames(vpath, start_step, n_frames)
        while len(frames) < n_frames:
            frames.append(frames[-1] if frames else np.zeros((IMG_H, IMG_W, 3), np.uint8))

        ep_name = f"episode_{ep_idx:06d}"
        write_video(frames, str(video_dir / f"{ep_name}.mp4"))

        rows = []
        for t in range(n_frames):
            row = ep_df.iloc[t]
            rows.append({
                "episode_index":    int(ep_idx),
                "frame_index":      int(t),
                "index":            global_frame_idx + t,
                "timestamp":        float(t) / FPS,
                "task_index":       0,
                "observation.state":[float(row[c]) for c in STATE_COLS],
                "action":           [float(row[c]) for c in ACTION_COLS],
            })

        ep_pq = pd.DataFrame(rows)
        ep_pq["timestamp"] = ep_pq["timestamp"].astype("float32")
        ep_pq["action"] = ep_pq["action"].apply(lambda x: [float(v) for v in x])
        ep_pq["observation.state"] = ep_pq["observation.state"].apply(lambda x: [float(v) for v in x])
        ep_pq.to_parquet(str(data_dir / f"{ep_name}.parquet"), index=False)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks":         [TASK_DESCRIPTION],
            "length":        n_frames,
            "dataset_from_index": 0,
            f"videos/{CAMERA_KEY}/chunk_index": 0,
            f"videos/{CAMERA_KEY}/file_index":  ep_idx,
        })
        global_frame_idx      += n_frames
        stats_actions_dfs.append(ep_df[ACTION_COLS])
        stats_states_dfs.append(ep_df[STATE_COLS])

    # meta/info.json
    n_eps = len(episodes_meta)
    info = {
        "codebase_version": "v2.0",
        "robot_type":        "franka",
        "total_episodes":    n_eps,
        "total_frames":      global_frame_idx,
        "total_tasks":       1,
        "total_videos":      n_eps,
        "total_chunks":      1,
        "chunks_size":       1000,
        "fps":               FPS,
        "splits":            {"train": f"0:{n_eps}"},
        "data_path":         "data/chunk-{chunk_index:03d}/{episode_index:06d}.parquet",
        "video_path":        "videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4",
        "features": {
            "action":            {"dtype": "float32", "shape": [len(ACTION_COLS)], "names": ACTION_COLS},
            "observation.state": {"dtype": "float32", "shape": [len(STATE_COLS)],  "names": STATE_COLS},
            CAMERA_KEY: {
                "dtype": "video", "shape": [IMG_H, IMG_W, 3],
                "names": ["height", "width", "channels"],
                "video_info": {"video.fps": float(FPS), "video.codec": "mp4v",
                               "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False},
            },
            "timestamp":     {"dtype": "float32", "shape": [1]},
            "frame_index":   {"dtype": "int64",   "shape": [1]},
            "episode_index": {"dtype": "int64",   "shape": [1]},
            "index":         {"dtype": "int64",   "shape": [1]},
            "task_index":    {"dtype": "int64",   "shape": [1]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")

    all_actions_df = pd.concat(stats_actions_dfs, ignore_index=True)
    all_states_df  = pd.concat(stats_states_dfs,  ignore_index=True)
    all_df = pd.concat([all_actions_df, all_states_df], axis=1)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(compute_stats(all_df), f, indent=2)

    print(f"\n=== LeRobot Dataset Created ===")
    print(f"Episodes: {len(episodes_meta)} | Frames: {global_frame_idx}")
    print(f"Output:   {os.path.abspath(OUTPUT_DIR)}/")
    print("Upload 'lerobot_dataset/' to Google Drive, then run the Colab notebook.")


if __name__ == "__main__":
    build_dataset()
