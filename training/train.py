#!/usr/bin/env python3
"""Shared real-PiPER training entry point. No simulator or external workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT
sys.path[:0] = [str(ROOT), str(SOURCE / 'openpi'), str(SOURCE / 'openpi/src'), str(SOURCE / 'openpi/packages/openpi-client/src')]

import yaml


def inside(value):
    path = Path(value).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path must be inside project_minimal: {path}')
    return str(path)


def load_params(path, robot=None, model=None, method=None, dataset=None):
    params = yaml.safe_load(Path(path).read_text())
    if not isinstance(params, dict):
        raise ValueError('Training YAML must be a mapping')
    for key, override, choices in [('robot', robot, ('single', 'bimanual')),
                                    ('model', model, ('pi0', 'pi05')),
                                    ('method', method, ('lora', 'full'))]:
        if override is not None:
            params[key] = override
        if params.get(key) not in choices:
            raise ValueError(f'{key} must be one of {choices}')
    params['use_lora'] = params['method'] == 'lora'
    if dataset:
        params.setdefault('data_collection', {})['dataset_repo_id'] = dataset if '/' in dataset else f'local/{dataset}'
    repo = params.get('data_collection', {}).get('dataset_repo_id', '')
    if len(repo.split('/')) != 2 or not repo.startswith('local/') or repo.split('/')[1] in ('', '.', '..'):
        raise ValueError('dataset_repo_id must be local/<name>')
    for key, default in [('model_cache_dir', 'models'), ('data_dir', 'datasets/lerobot'),
                         ('output_dir', 'checkpoints'), ('assets_dir', 'assets')]:
        params[key] = inside(params.get(key) or default)
    params['data_collection']['save_dir'] = params['data_dir']
    if params.get('overwrite', False):
        raise ValueError('Shared trainer disables overwrite; use a new experiment or resume: true')
    if params.get('base_params_path'):
        base = params['base_params_path']
        if '://' not in base:
            params['base_params_path'] = inside(base)
    else:
        params['base_params_path'] = f"gs://openpi-assets/checkpoints/{params['model']}_base/params"
    return params


def build_config(params):
    devices = params.get('cuda_visible_devices')
    if devices is not None:
        if not isinstance(devices, str) or not devices or any(not part.isdecimal() for part in devices.split(',')):
            raise ValueError('cuda_visible_devices must be null or comma-separated GPU indices, e.g. "0,1"')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices
    from training.config import build_train_config
    return build_train_config(params)


def check_dataset(cfg, robot):
    if robot == 'bimanual':
        from training.bimanual import check_dataset as check
        return check(cfg)
    import numpy as np
    import pyarrow.parquet as pq
    root = Path(os.environ['HF_LEROBOT_HOME']) / cfg.data.repo_id
    info = json.loads((root / 'meta/info.json').read_text())
    names = [f'joint_{i}' for i in range(1, 7)] + ['gripper']
    if info.get('codebase_version') != 'v2.1' or info.get('robot_type') != 'piper_single_arm':
        raise ValueError('Expected single-arm LeRobot v2.1 data')
    for key in ('observation.state', 'action'):
        feature = info['features'][key]
        if feature.get('shape') != [7] or feature.get('names') != names or feature.get('dtype') != 'float32':
            raise ValueError(f'Expected 7D recorded joint/gripper values: {key}')
    cameras = {k for k in info['features'] if k.startswith('observation.images.')}
    if cameras != {'observation.images.front', 'observation.images.wrist'}:
        raise ValueError('Expected front and wrist cameras')
    episodes = [json.loads(s) for s in (root / 'meta/episodes.jsonl').read_text().splitlines() if s]
    tasks = [json.loads(s) for s in (root / 'meta/tasks.jsonl').read_text().splitlines() if s]
    if not tasks or any(not t['task'].strip() for t in tasks):
        raise ValueError('Dataset requires nonempty task prompts')
    total = 0
    for ep in episodes:
        idx = ep['episode_index']
        values = dict(episode_index=idx, episode_chunk=idx // info['chunks_size'])
        table = pq.read_table(root / info['data_path'].format(**values))
        for key in ('observation.state', 'action'):
            array = np.asarray(table[key].to_pylist())
            if array.shape != (ep['length'], 7) or not np.isfinite(array).all():
                raise ValueError(f'Invalid {key} values')
        times = np.asarray(table['timestamp'].to_pylist())
        if not np.isfinite(times).all() or abs(times[0]) > 1e-4 or np.any(abs(np.diff(times) - 1/info['fps']) > 1e-4):
            raise ValueError('Invalid fixed-rate timestamps')
        for camera in cameras:
            if not (root / info['video_path'].format(**values, video_key=camera)).is_file():
                raise FileNotFoundError(f'Missing episode video: {camera}')
        total += len(times)
    if total < cfg.batch_size or total != info['total_frames'] or len(episodes) != info['total_episodes']:
        raise ValueError('Dataset frame/episode count mismatch or too few samples')
    return dict(root=str(root), episodes=len(episodes), frames=total)


def ensure_norm_stats(cfg, params, recompute=False):
    from scripts import compute_norm_stats
    root = Path(os.environ['HF_LEROBOT_HOME']) / cfg.data.repo_id
    signature = dict(version=1, repo_id=cfg.data.repo_id, root=str(root.resolve()),
                     info_sha256=hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest(),
                     episodes_sha256=hashlib.sha256((root / 'meta/episodes.jsonl').read_bytes()).hexdigest(),
                     robot=params['robot'], model=params['model'], action_horizon=cfg.model.action_horizon,
                     batch_size=cfg.batch_size, action_format='delta_joint_absolute_gripper',
                     sample_validity_key='quality.frame_valid' if params['robot']=='bimanual' else None)
    folder = cfg.assets_dirs / cfg.data.repo_id
    record = folder / 'tuning_norm_config.json'
    stats = folder / 'norm_stats.json'
    if cfg.resume and recompute:
        raise ValueError('Cannot recompute statistics while resuming')
    if stats.is_file() and not recompute:
        if not record.is_file() or json.loads(record.read_text()) != signature:
            raise ValueError('Normalization configuration changed; recompute explicitly for a new run')
        return stats
    if cfg.resume:
        raise ValueError('Resume requires existing matching normalization statistics')
    compute_norm_stats.main(config_name=cfg.name)
    if not stats.is_file():
        raise FileNotFoundError(stats)
    record.write_text(json.dumps(signature, indent=2) + '\n')
    return stats


def validate_batch(cfg, robot):
    import numpy as np
    from openpi import transforms
    from openpi.training import data_loader
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model)
    first = raw[0]
    dim = 7 if robot == 'single' else 14
    state = np.asarray(first['observation.state']).copy()
    original_actions = np.asarray(first['action']).copy()
    if state.shape != (dim,) or original_actions.shape != (cfg.model.action_horizon, dim):
        raise ValueError('Raw state/action dimensions differ from robot configuration')
    sample = first
    for transform in (*data.repack_transforms.inputs, *data.data_transforms.inputs):
        sample = transform(sample)
    joints = list(range(6)) if robot == 'single' else [*range(6), *range(7,13)]
    expected = original_actions.copy()
    expected[:, joints] -= state[joints]
    np.testing.assert_array_equal(sample['actions'], expected)
    np.testing.assert_array_equal(sample['state'], state)
    loader = data_loader.create_torch_data_loader(data_config=data, model_config=cfg.model,
        action_horizon=cfg.model.action_horizon, batch_size=cfg.batch_size, skip_norm_stats=data.norm_stats is None,
        num_batches=1, num_workers=0)
    obs, actions = next(iter(loader))
    if actions.shape != (cfg.batch_size, cfg.model.action_horizon, 32) or obs.state.shape != (cfg.batch_size, 32):
        raise ValueError('Invalid padded batch shapes')
    if not np.isfinite(actions).all() or not np.isfinite(obs.state).all():
        raise ValueError('Nonfinite model inputs')
    np.testing.assert_array_equal(np.asarray(actions)[..., dim:], 0)
    np.testing.assert_array_equal(np.asarray(obs.state)[..., dim:], 0)
    if set(obs.images) != {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}:
        raise ValueError('Unexpected model camera inputs')
    for value in obs.images.values():
        if value.shape != (cfg.batch_size,224,224,3) or not np.isfinite(value).all():
            raise ValueError('Invalid image batch')
    restored = transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm)(
        {'state':np.asarray(obs.state[0]).copy(), 'actions':np.asarray(actions[0]).copy()})
    for transform in data.data_transforms.outputs:
        restored = transform(restored)
    np.testing.assert_allclose(restored['actions'], original_actions, rtol=1e-5, atol=1e-6)
    return dict(state_shape=list(obs.state.shape), action_shape=list(actions.shape),
                normalization_checked=data.norm_stats is not None, joint_delta_verified=True,
                absolute_action_roundtrip_verified=True)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', choices=['single', 'bimanual'])
    parser.add_argument('--model', choices=['pi0', 'pi05'])
    parser.add_argument('--method', choices=['lora', 'full'])
    parser.add_argument('--dataset')
    parser.add_argument('--config', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-config', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    modes.add_argument('--norm-stats-only', action='store_true')
    parser.add_argument('--recompute-norm-stats', action='store_true')
    args = parser.parse_args(argv)
    if not args.config and not all((args.robot, args.model, args.method)):
        parser.error('Specify --config or --robot, --model and --method')
    path = args.config or ROOT / 'configs' / args.robot / f'{args.model}_{args.method}.yaml'
    params = load_params(path, args.robot, args.model, args.method, args.dataset)
    cfg = build_config(params)
    print(json.dumps(dict(config=cfg.name, model=cfg.model.model_type.value, dataset=cfg.data.repo_id,
        checkpoint_dir=str(cfg.checkpoint_dir), base_params=params['base_params_path'],
        language_variant=cfg.model.paligemma_variant, action_variant=cfg.model.action_expert_variant,
        lora_rank=cfg.model.lora_rank, horizon=cfg.model.action_horizon, batch_size=cfg.batch_size,
        resume=cfg.resume, freeze_filter=repr(cfg.freeze_filter)), indent=2), flush=True)
    if args.check_config:
        return 0
    if not args.validate_only and not args.norm_stats_only:
        import jax
        count = jax.device_count()
        if jax.default_backend() != 'gpu':
            raise RuntimeError('Training requires a recognized NVIDIA GPU; check nvidia-smi and Compose GPU configuration')
        if cfg.batch_size % count or count % cfg.fsdp_devices:
            raise ValueError(f'batch_size ({cfg.batch_size}) must be divisible by visible GPUs ({count}), '
                             f'and visible GPUs must be divisible by fsdp_devices ({cfg.fsdp_devices}). '
                             'Set cuda_visible_devices in YAML or adjust batch_size/fsdp_devices.')
    print(json.dumps(check_dataset(cfg, params['robot']), indent=2), flush=True)
    if args.validate_only:
        print(json.dumps(validate_batch(cfg, params['robot']), indent=2))
        return 0
    if cfg.resume and not cfg.checkpoint_dir.is_dir():
        raise FileNotFoundError('Resume checkpoint directory does not exist')
    stats = ensure_norm_stats(cfg, params, args.recompute_norm_stats)
    if args.norm_stats_only:
        return 0
    # Keep complete effective configuration separate from Orbax's checkpoint directory.
    run_dir = ROOT / 'logs' / cfg.name / cfg.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    effective = run_dir / 'training.yaml'
    if effective.exists() and not cfg.resume:
        raise FileExistsError(f'Experiment metadata already exists: {effective}; choose another exp_name')
    if cfg.resume:
        if not effective.is_file():
            raise FileNotFoundError('Original training.yaml is required for resume')
        previous = yaml.safe_load(effective.read_text())
        mutable = {'resume', 'num_train_steps', 'no_wandb'}
        if {k:v for k,v in previous.items() if k not in mutable} != {k:v for k,v in params.items() if k not in mutable}:
            raise ValueError('Resume configuration differs from original training')
    else:
        effective.write_text(yaml.safe_dump(params, sort_keys=False))
    metadata = dict(config=cfg.name, experiment=cfg.exp_name, robot=params['robot'], model=params['model'],
                    method=params['method'], base_params=params['base_params_path'],
                    checkpoint_dir=str(cfg.checkpoint_dir), norm_stats=str(stats), training_yaml=str(effective),
                    weights='OpenPI Orbax full parameter tree (including LoRA when enabled); not adapter-only')
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    logging.getLogger().addHandler(logging.FileHandler(run_dir / 'train.log'))
    from training.runtime import load_trainer
    load_trainer().main(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
