"""Dataset checks copied from the verified dual-PiPER trainer."""
import json
import os
from pathlib import Path

CAMERAS = ("front", "left_wrist", "right_wrist")
ACTION_FORMAT = "delta_joint_absolute_gripper"
DELTA_MASK = [True] * 6 + [False] + [True] * 6 + [False]
VALUE_NAMES = [
    name for arm in ("left", "right") for name in [*(f"{arm}_joint_{i}_rad" for i in range(1, 7)), f"{arm}_gripper_m"]
]


def dataset_path(cfg):
    return Path(os.environ["HF_LEROBOT_HOME"]) / cfg.data.repo_id


def check_dataset(cfg):
    """Read metadata and scalar quality columns; never operate on partial outputs."""
    import numpy as np
    import pyarrow.parquet as pq

    root = dataset_path(cfg)
    if not (root / "meta/info.json").is_file():
        partials = sorted(root.parent.glob(f".{root.name}.partial-*"))
        raise FileNotFoundError(
            f"Completed dataset missing: {root}. Partial outputs: {partials}. "
            "Wait for conversion; no partial directory will be moved or edited."
        )
    info = json.loads((root / "meta/info.json").read_text())
    provenance_path = root / "provenance/conversion.json"
    if not provenance_path.is_file() or json.loads(provenance_path.read_text()).get("status") != "verified":
        raise ValueError("Dual-PiPER conversion has no successful final verification record")
    if info.get("codebase_version") != "v2.1" or info.get("fps") != 20 or info.get("robot_type") != "piper_bimanual":
        raise ValueError("Expected piper_bimanual LeRobot v2.1 at 20Hz")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature.get("shape") != [14] or feature.get("dtype") != "float32" or feature.get("names") != VALUE_NAMES:
            raise ValueError(f"Wrong 14D left-to-right feature: {key}")
    cameras = {k for k in info["features"] if k.startswith("observation.images.")}
    if cameras != {f"observation.images.{c}" for c in CAMERAS}:
        raise ValueError("Expected front/left_wrist/right_wrist camera keys")
    if info["features"].get("quality.frame_valid", {}).get("dtype") != "bool":
        raise ValueError("quality.frame_valid must be bool")
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line]
    if len(episodes) != info["total_episodes"] or not tasks or any(not t["task"].strip() for t in tasks):
        raise ValueError("Episode/task metadata mismatch")
    valid_count = total = 0
    for ep in episodes:
        idx = ep["episode_index"]
        values = {"episode_index": idx, "episode_chunk": idx // info["chunks_size"]}
        table = pq.read_table(root / info["data_path"].format(**values), columns=["quality.frame_valid", "timestamp"])
        valid = np.asarray(table["quality.frame_valid"].to_pylist())
        times = np.asarray(table["timestamp"].to_pylist())
        if valid.dtype != np.bool_ or valid.ndim != 1 or len(valid) != ep["length"]:
            raise ValueError("Invalid quality column or episode length")
        if not np.isfinite(times).all() or abs(times[0]) > 1e-4 or np.any(abs(np.diff(times) - 0.05) > 1e-4):
            raise ValueError("Episode is not an ordered 20Hz grid")
        for camera in cameras:
            video = root / info["video_path"].format(**values, video_key=camera)
            if not video.is_file():
                raise FileNotFoundError(video)
        total += len(valid)
        valid_count += int(valid.sum())
    if total != info["total_frames"] or valid_count < cfg.batch_size:
        raise ValueError("Frame count mismatch or fewer valid anchors than one batch")
    return {
        "root": str(root),
        "episodes": len(episodes),
        "frames": total,
        "valid_anchors": valid_count,
        "invalid_anchors": total - valid_count,
        "tasks": [t["task"] for t in tasks],
    }

