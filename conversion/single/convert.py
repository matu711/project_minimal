#!/usr/bin/env python3
"""Convert extracted, synchronized PiPER demonstrations to LeRobotDataset.

This is the second conversion stage; it does not decode an MCAP container.
The first stage must decode/resample the robot-specific MCAP topics into the
raw layout below. Topic names and ROS message types are intentionally not
guessed because they must match the real recorder.

Expected raw layout (piper_raw_fixed_rate_synced_v5):

    piper_raw_white_pickup/
      dataset_meta.json
      20260825_224317/
        data.npz
        meta.json
        front/000000.jpg ...
        wrist/000000.jpg ...
      episode_000001/
        ...

Output features:
    observation.images.front : RGB camera image
    observation.images.wrist : RGB wrist camera image
    observation.state        : follower [joint_1..joint_6, gripper]
    observation.ee_pose      : optional [x, y, z, qx, qy, qz, qw, gripper]
    action                   : leader   [joint_1..joint_6, gripper]
    task                     : natural-language task stored by LeRobot

The state/action values are kept as absolute joint/gripper positions from the
raw data. They are NOT converted to Cartesian/OSC deltas.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_VALUE_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
SUPPORTED_RAW_FORMATS = {"piper_raw_v1", "piper_raw_fixed_rate_synced_v5", "piper_raw_fixed_rate_synced_v6_ee_pose"}


from conversion.common.image_utils import load_rgb
from conversion.common.fps import resolve_fps
from conversion.common.task import resolve_task


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extracted PiPER demonstrations to LeRobotDataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        dest="inputs",
        required=True,
        help=(
            "Extracted raw dataset directory containing dataset_meta.json and episodes. "
            "Repeat --input to merge multiple raw datasets into one LeRobot dataset."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the LeRobot dataset. If omitted, infer from --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional fine-tuning YAML config. When provided, the converter "
            "uses data_collection.save_dir and data_collection.dataset_repo_id "
            "to choose the output path automatically."
        ),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="LeRobot repo_id. Defaults to --config value, then dataset_meta.json value.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override task text for every episode in every input dataset.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("joint_absolute", "ee_delta_axis_angle"),
        default="joint_absolute",
        help=(
            "Action representation. joint_absolute preserves recorded joint commands; "
            "ee_delta_axis_angle derives base-frame EE deltas from consecutive corrected poses."
        ),
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional image resize, e.g. --resize 224 224. Default: preserve raw size.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Also convert episodes whose meta.json has successful=false.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the raw dataset only; do not import LeRobot or write output.",
    )
    parser.add_argument("--timing-log", type=Path, help="Save actual raw clocks and nominal-time warnings to a new JSON log")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def correct_quaternion_continuity(ee_poses: np.ndarray) -> tuple[np.ndarray, int]:
    """Normalize xyzw quaternions and remove q/-q sign jumps within one episode."""
    corrected = np.asarray(ee_poses, dtype=np.float32).copy()
    quaternions = corrected[:, 3:7].astype(np.float64)
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-8):
        bad_index = int(np.flatnonzero(norms < 1e-8)[0])
        raise ValueError(f"Zero-length quaternion at frame {bad_index}")
    quaternions /= norms[:, None]

    flips = 0
    for i in range(1, len(quaternions)):
        if float(np.dot(quaternions[i - 1], quaternions[i])) < 0.0:
            quaternions[i] *= -1.0
            flips += 1

    corrected[:, 3:7] = quaternions.astype(np.float32)
    return corrected, flips


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as [x, y, z, w]."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def quaternion_delta_rotvec(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return shortest-path base/world-frame rotation delta in axis-angle radians."""
    current_inverse = np.asarray([-current[0], -current[1], -current[2], current[3]])
    relative = quaternion_multiply_xyzw(target, current_inverse)
    relative /= np.linalg.norm(relative)
    if relative[3] < 0.0:
        relative *= -1.0
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm < 1e-8:
        return (2.0 * relative[:3]).astype(np.float32)
    angle = 2.0 * np.arctan2(vector_norm, float(relative[3]))
    return (relative[:3] / vector_norm * angle).astype(np.float32)


def make_ee_delta_actions(ee_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Create [dxyz, base-frame rotation-vector, absolute gripper] targets."""
    corrected, flips = correct_quaternion_continuity(ee_poses)
    if len(corrected) < 2:
        raise ValueError("At least two EE poses are required for delta actions")
    actions = np.empty((len(corrected) - 1, 7), dtype=np.float32)
    actions[:, :3] = corrected[1:, :3] - corrected[:-1, :3]
    for i in range(len(actions)):
        actions[i, 3:6] = quaternion_delta_rotvec(corrected[i, 3:7], corrected[i + 1, 3:7])
    actions[:, 6] = corrected[1:, 7]
    return corrected, actions, flips


def episode_dirs(raw_root: Path) -> list[Path]:
    episodes = sorted(p for p in raw_root.iterdir() if p.is_dir() and (p / "data.npz").is_file() and (p / "meta.json").is_file())
    if not episodes:
        raise RuntimeError(f"No episode directories with data.npz/meta.json found under: {raw_root}")
    return episodes


def validate_episode(
    ep_dir: Path,
    state_dim: int,
    resize: tuple[int, int] | None,
    ee_dim: int | None = None,
) -> tuple[int, tuple[int, int, int]]:
    npz_path = ep_dir / "data.npz"
    meta_path = ep_dir / "meta.json"

    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing: {npz_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing: {meta_path}")

    with np.load(npz_path) as data:
        required = {"observation_state", "action"}
        if ee_dim is not None:
            required.add("observation_ee")
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{ep_dir.name}: data.npz missing keys: {sorted(missing)}")

        states = data["observation_state"]
        actions = data["action"]

        if states.ndim != 2 or states.shape[1] != state_dim:
            raise ValueError(
                f"{ep_dir.name}: observation_state shape {states.shape}, expected (N, {state_dim})"
            )
        if actions.ndim != 2 or actions.shape[1] != state_dim:
            raise ValueError(
                f"{ep_dir.name}: action shape {actions.shape}, expected (N, {state_dim})"
            )
        if len(states) != len(actions):
            raise ValueError(
                f"{ep_dir.name}: state/action length mismatch: {len(states)} vs {len(actions)}"
            )
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"{ep_dir.name}: state/action contains NaN or infinity")

        if ee_dim is not None:
            ee_pose = data["observation_ee"]
            if ee_pose.ndim != 2 or ee_pose.shape != (len(states), ee_dim):
                raise ValueError(
                    f"{ep_dir.name}: observation_ee shape {ee_pose.shape}, "
                    f"expected ({len(states)}, {ee_dim})"
                )
            if not np.isfinite(ee_pose).all():
                raise ValueError(f"{ep_dir.name}: observation_ee contains NaN or infinity")

        n = len(states)

    front_files = sorted((ep_dir / "front").glob("*.jpg"))
    wrist_files = sorted((ep_dir / "wrist").glob("*.jpg"))
    if len(front_files) != n or len(wrist_files) != n:
        raise ValueError(
            f"{ep_dir.name}: expected {n} images, got front={len(front_files)}, wrist={len(wrist_files)}"
        )

    # Verify exact frame filenames, not only file counts.
    for i in range(n):
        if not (ep_dir / "front" / f"{i:06d}.jpg").is_file():
            raise FileNotFoundError(f"{ep_dir.name}: missing front/{i:06d}.jpg")
        if not (ep_dir / "wrist" / f"{i:06d}.jpg").is_file():
            raise FileNotFoundError(f"{ep_dir.name}: missing wrist/{i:06d}.jpg")

    sample = load_rgb(ep_dir / "front" / "000000.jpg", resize)
    return n, tuple(sample.shape)


def import_lerobot_dataset():
    """Support both common LeRobot import paths."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            try:
                from lerobot.datasets import LeRobotDataset
            except ImportError as exc:
                raise RuntimeError(
                    "LeRobot is not installed in this Python environment. "
                    "Run this script in the environment where your OpenPI/LeRobot package is installed."
                ) from exc
    return LeRobotDataset


def infer_output_root(
    *,
    explicit_output: Path | None,
    explicit_repo_id: str | None,
    config_path: Path | None,
    dataset_meta: dict[str, Any],
) -> tuple[Path, str]:
    config_repo_id: str | None = None
    config_output_root: Path | None = None

    if config_path is not None:
        cfg = load_yaml(config_path)
        data_collection = cfg.get("data_collection") or {}
        if not isinstance(data_collection, dict):
            raise ValueError(f"data_collection must be a mapping in {config_path}")
        config_repo_id = data_collection.get("dataset_repo_id")
        save_dir = data_collection.get("save_dir") or cfg.get("data_dir")
        config_output_root = resolve_path(save_dir)

    repo_id = (
        explicit_repo_id
        or config_repo_id
        or dataset_meta.get("repo_id")
        or "local/piper_pickup"
    )

    if explicit_output is not None:
        return explicit_output.expanduser().resolve(), repo_id

    if config_output_root is not None:
        return (config_output_root / repo_id).resolve(), repo_id

    raise SystemExit(
        "--output is required unless you pass --config with "
        "data_collection.save_dir and data_collection.dataset_repo_id."
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    raw_roots = [path.expanduser().resolve() for path in args.inputs]

    input_specs: list[tuple[Path, dict[str, Any]]] = []
    for raw_root in raw_roots:
        dataset_meta_path = raw_root / "dataset_meta.json"
        if not dataset_meta_path.is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {dataset_meta_path}")
        metadata = load_json(dataset_meta_path)
        episode_metadata = [load_json(p / "meta.json") for p in episode_dirs(raw_root)]
        episode_metadata = [m for m in episode_metadata if args.include_failed or m.get("successful") is not False]
        metadata["fps"] = resolve_fps(metadata, episode_metadata, str(raw_root))
        raw_format = metadata.get("format")
        if raw_format not in SUPPORTED_RAW_FORMATS:
            raise ValueError(
                f"Unsupported raw format in {raw_root}: {raw_format!r}; "
                f"expected one of {sorted(SUPPORTED_RAW_FORMATS)}"
            )
        input_specs.append((raw_root, metadata))

    dataset_meta = input_specs[0][1]
    fps = int(dataset_meta.get("fps", 20))
    state_dim = int(dataset_meta.get("state_dim", 7))
    value_names = list(dataset_meta.get("value_names", DEFAULT_VALUE_NAMES))
    if len(value_names) != state_dim:
        raise ValueError(
            f"value_names has {len(value_names)} entries but state_dim={state_dim}"
        )
    ee_dim_value = dataset_meta.get("ee_observation_dim")
    ee_dim = int(ee_dim_value) if ee_dim_value is not None else None
    ee_names = list(dataset_meta.get("observation_ee_layout", []))
    if ee_dim is not None and len(ee_names) != ee_dim:
        raise ValueError(
            f"observation_ee_layout has {len(ee_names)} entries but "
            f"ee_observation_dim={ee_dim}"
        )
    robot_type = dataset_meta.get("robot_type", "piper_single_arm")

    expected_schema = {
        "fps": fps,
        "state_dim": state_dim,
        "value_names": value_names,
        "ee_observation_dim": ee_dim,
        "observation_ee_layout": ee_names,
        "robot_type": robot_type,
    }
    for raw_root, metadata in input_specs[1:]:
        actual_schema = {
            "fps": int(metadata.get("fps", 20)),
            "state_dim": int(metadata.get("state_dim", 7)),
            "value_names": list(metadata.get("value_names", DEFAULT_VALUE_NAMES)),
            "ee_observation_dim": (
                int(metadata["ee_observation_dim"])
                if metadata.get("ee_observation_dim") is not None
                else None
            ),
            "observation_ee_layout": list(metadata.get("observation_ee_layout", [])),
            "robot_type": metadata.get("robot_type", "piper_single_arm"),
        }
        mismatches = {
            key: (expected_schema[key], actual_schema[key])
            for key in expected_schema
            if expected_schema[key] != actual_schema[key]
        }
        if mismatches:
            raise ValueError(f"Incompatible dataset schema in {raw_root}: {mismatches}")

    if args.action_mode == "ee_delta_axis_angle" and ee_dim != 8:
        raise ValueError(
            "--action-mode ee_delta_axis_angle requires an 8D observation_ee "
            "[x, y, z, qx, qy, qz, qw, gripper]"
        )
    if args.action_mode == "ee_delta_axis_angle":
        action_names = [
            "delta_x_m",
            "delta_y_m",
            "delta_z_m",
            "delta_rx_rad",
            "delta_ry_rad",
            "delta_rz_rad",
            "gripper_m",
        ]
    else:
        action_names = value_names

    output_root, repo_id = infer_output_root(
        explicit_output=args.output,
        explicit_repo_id=args.repo_id,
        config_path=resolve_path(args.config),
        dataset_meta=dataset_meta,
    )
    resize: tuple[int, int] | None = None
    if args.resize is not None:
        width, height = args.resize
        if width <= 0 or height <= 0:
            raise ValueError("--resize WIDTH HEIGHT must both be positive")
        resize = (width, height)

    selected: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    input_summaries: list[tuple[Path, int, int]] = []
    skipped_failed = 0
    total_input_episodes = 0

    for raw_root, metadata in input_specs:
        all_eps = episode_dirs(raw_root)
        total_input_episodes += len(all_eps)
        selected_before = len(selected)
        for ep_dir in all_eps:
            ep_meta = load_json(ep_dir / "meta.json")
            if not args.include_failed and ep_meta.get("successful") is False:
                skipped_failed += 1
                continue
            selected.append((ep_dir, ep_meta, metadata))
        input_summaries.append((raw_root, len(all_eps), len(selected) - selected_before))

    if not selected:
        raise RuntimeError("No episodes selected for conversion")

    task_counts = Counter(
        resolve_task(args.task, ep_meta, metadata, "single raw")
        for _, ep_meta, metadata in selected
    )

    total_frames = 0
    image_shape: tuple[int, int, int] | None = None

    print(f"Raw datasets: {len(input_specs)}")
    for raw_root, input_count, selected_count in input_summaries:
        print(f"  {raw_root}: {selected_count} selected / {input_count} total")
    if args.config is not None:
        print(f"Train config: {resolve_path(args.config)}")
    print(f"Episodes    : {len(selected)} selected / {total_input_episodes} total")
    if skipped_failed:
        print(f"Skipped     : {skipped_failed} failed episode(s)")
    print("Tasks       :")
    for task, count in sorted(task_counts.items()):
        print(f"  {count:4d} episode(s): {task}")
    print(f"FPS         : {fps}")
    print(f"State/action: {state_dim} dims = {value_names}")
    if ee_dim is not None:
        print(f"EE pose     : {ee_dim} dims = {ee_names}")
    print(f"repo_id     : {repo_id}")
    print(f"Output path : {output_root}")
    print(f"Action mode : {args.action_mode}")

    quaternion_flips = 0
    for ep_dir, _, _ in selected:
        n, shape = validate_episode(
            ep_dir, state_dim=state_dim, resize=resize, ee_dim=ee_dim
        )
        total_frames += n
        if ee_dim is not None:
            with np.load(ep_dir / "data.npz") as data:
                ee_poses = np.asarray(data["observation_ee"], dtype=np.float32)
            if args.action_mode == "ee_delta_axis_angle":
                _, _, flips = make_ee_delta_actions(ee_poses)
            else:
                _, flips = correct_quaternion_continuity(ee_poses)
            quaternion_flips += flips
        if image_shape is None:
            image_shape = shape
        elif shape != image_shape:
            raise ValueError(
                f"Image shape mismatch: {ep_dir.name} has {shape}, expected {image_shape}"
            )
        print(f"  validated {ep_dir.name}: {n} frames")

    assert image_shape is not None
    h, w, c = image_shape
    if c != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got shape {image_shape}")

    print(f"Total frames: {total_frames}")
    print(f"Image shape : {image_shape} (H, W, C)")
    if ee_dim is not None:
        print(f"Quaternion continuity flips: {quaternion_flips}")
    if args.action_mode == "ee_delta_axis_angle":
        print(f"Output frames: {total_frames - len(selected)} (last frame dropped per episode)")

    from conversion.common.timing_diagnostics import record_timing
    timing_report = record_timing([p for p, _, _ in selected], fps, output_root, args.timing_log)

    if args.validate_only:
        print("Validation OK. No LeRobot dataset was written.")
        return

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}\n"
                "Use --overwrite if you really want to replace it."
            )
        shutil.rmtree(output_root)

    LeRobotDataset = import_lerobot_dataset()

    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": value_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }
    if ee_dim is not None:
        features["observation.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_dim,),
            "names": ee_names,
        }

    print(f"\nCreating LeRobot dataset: {output_root}")
    print(f"repo_id     : {repo_id}")
    print(f"robot_type  : {robot_type}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=True,
    )

    timing_folder = output_root / "provenance"
    timing_folder.mkdir(exist_ok=True)
    (timing_folder / "raw_timing.json").write_text(json.dumps(timing_report, indent=2, allow_nan=False) + "\n")

    written_frames = 0

    try:
        for out_ep_index, (ep_dir, ep_meta, metadata) in enumerate(selected):
            with np.load(ep_dir / "data.npz") as data:
                states = np.asarray(data["observation_state"], dtype=np.float32)
                recorded_actions = np.asarray(data["action"], dtype=np.float32)
                raw_ee_poses = (
                    np.asarray(data["observation_ee"], dtype=np.float32)
                    if ee_dim is not None
                    else None
                )

            episode_flips = 0
            if args.action_mode == "ee_delta_axis_angle":
                assert raw_ee_poses is not None
                corrected_ee_poses, actions, episode_flips = make_ee_delta_actions(raw_ee_poses)
                states = states[:-1]
                ee_poses = corrected_ee_poses[:-1]
            else:
                actions = recorded_actions
                if raw_ee_poses is not None:
                    ee_poses, episode_flips = correct_quaternion_continuity(raw_ee_poses)
                else:
                    ee_poses = None

            task = resolve_task(args.task, ep_meta, metadata, "single raw")
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError(f"{ep_dir.name}: state/action contains NaN or infinity")
            if len(states) != len(actions):
                raise ValueError(
                    f"{ep_dir.name}: converted state/action length mismatch: "
                    f"{len(states)} vs {len(actions)}"
                )

            n = len(states)

            for i in range(n):
                front = load_rgb(ep_dir / "front" / f"{i:06d}.jpg", resize)
                wrist = load_rgb(ep_dir / "wrist" / f"{i:06d}.jpg", resize)

                frame = {
                    "observation.images.front": front,
                    "observation.images.wrist": wrist,
                    "observation.state": states[i],
                    "action": actions[i],
                    "task": task,
                }
                if ee_poses is not None:
                    frame["observation.ee_pose"] = ee_poses[i]

                # Deliberately do not pass the irregular wall-clock timestamp from
                # the raw recorder. LeRobot will assign frame_index / fps, keeping
                # the episode on the nominal 20 Hz timeline used for training.
                dataset.add_frame(frame)
                written_frames += 1

            dataset.save_episode()
            print(
                f"  saved episode {out_ep_index:06d} "
                f"<- {ep_dir.name} ({n} frames, task={task!r})"
            )

    finally:
        # Required for LeRobotDataset v3; harmlessly skipped on older versions.
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            finalize()

    print("\nDone.")
    print(f"Output      : {output_root}")
    print(f"Episodes    : {len(selected)}")
    print(f"Frames      : {written_frames}")
    feature_names = [
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
    ]
    if ee_dim is not None:
        feature_names.append("observation.ee_pose")
    feature_names.extend(["action", "task"])
    print(f"Features    : {', '.join(feature_names)}")


if __name__ == "__main__":
    main()
