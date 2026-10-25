"""Read synchronized recorder bundles; recorded row/image correspondence is retained, never recovered."""
import csv
import json
from pathlib import Path
import numpy as np
from conversion.common.episode import Episode
from conversion.common.fps import resolve_fps
from conversion.common.task import resolve_task


def raw_episodes(inputs, spec):
    found = []
    for root in inputs:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Raw input directory not found: {root}')
        paths = [root] if (root/'data.npz').is_file() else sorted(
            p for p in root.iterdir() if p.is_dir() and ((p/'data.npz').exists() or (p/'meta.json').exists()))
        dataset_path = root/'dataset_meta.json' if (root/'dataset_meta.json').exists() else root.parent/'dataset_meta.json'
        dataset = json.loads(dataset_path.read_text()) if dataset_path.exists() else {}
        for path in paths:
            if not (path/'meta.json').is_file() or not (path/'data.npz').is_file():
                raise ValueError(f'{path}: missing meta.json or data.npz')
            meta = json.loads((path/'meta.json').read_text())
            if spec.name == 'single' and meta.get('successful') is False:
                continue
            found.append((path.resolve(), dataset, meta))
    if not found or len({p for p, _, _ in found}) != len(found):
        raise ValueError('No usable raw episodes or duplicate raw inputs')
    return found


def read_episode(path, dataset, meta, spec, task, timings):
    with timings.measure('raw_read'):
        for info in (dataset, meta):
            if spec.name == 'single' and info.get('value_names', list(spec.value_names)) != list(spec.value_names):
                raise ValueError(f'{path}: unexpected single-arm joint/gripper order')
            if info.get('state_dim', spec.dim) != spec.dim or info.get('action_dim', spec.dim) != spec.dim:
                raise ValueError(f'{path}: wrong raw state/action dimension')
            for key, unit in [('joint_unit', 'rad'), ('gripper_unit', 'm')]:
                if info.get(key, unit) != unit:
                    raise ValueError(f'{path}: wrong {key}; no implicit unit conversion')
            if 'units' in info and any(info['units'].get(k) != v for k, v in [('joints','rad'),('gripper','m')]):
                raise ValueError(f'{path}: raw units must be rad/m')
        task = resolve_task(task, meta, dataset, str(path))
        with np.load(path/'data.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        for key in ('observation_state', 'action'):
            if key not in arrays:
                raise ValueError(f'{path}: missing NPZ field {key}')
        states = np.asarray(arrays['observation_state'], np.float32)
        actions = np.asarray(arrays['action'], np.float32)
        if states.ndim != 2 or states.shape[1] != spec.dim or actions.shape != states.shape or not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f'{path}: State/Action must be finite (N,{spec.dim}) arrays')
        count = len(states)
        if meta.get('num_frames', count) != count:
            raise ValueError(f'{path}: metadata frame count differs from NPZ')
        if spec.name == 'bimanual':
            expected = [f'{arm}_{joint}' for arm in ('right','left') for joint in [*(f'joint{i}' for i in range(1,7)), 'gripper']]
            if meta.get('names', expected) != expected:
                raise ValueError(f'{path}: expected recorded right-seven then left-seven order')
            if states.ndim != 2 or states.shape[1] != 14 or actions.shape != states.shape:
                raise ValueError(f'{path}: expected State/Action shape (N,14)')
            order = [*range(7,14), *range(7)]
            states, actions = states[:,order], actions[:,order]
        rows = None
        if (path/'frames.csv').is_file():
            with (path/'frames.csv').open(newline='') as file:
                rows = list(csv.DictReader(file))
            required = {'frame_index', *(c+'_image' for c in spec.cameras)}
            if rows and not required.issubset(rows[0]):
                raise ValueError(f'{path}: missing CSV columns {sorted(required-set(rows[0]))}')
            if len(rows) != count:
                raise ValueError(f'{path}: CSV/NPZ frame count mismatch')
        elif spec.name == 'bimanual':
            raise ValueError(f'{path}: missing frames.csv')
        indices = np.asarray([int(row['frame_index']) for row in rows]) if rows else arrays.get('frame_index')
        if indices is not None and 'frame_index' in arrays and not np.array_equal(indices, arrays['frame_index']):
            raise ValueError(f'{path}: CSV/NPZ frame IDs disagree')
        images = {}
        for camera, out in spec.cameras.items():
            names = [row[f'{camera}_image'] for row in rows] if rows else [f'{camera}/{i:06d}.jpg' for i in range(count)]
            files = [(path/name).resolve() for name in names]
            if any(not f.is_relative_to(path.resolve()) or not f.is_file() for f in files):
                raise ValueError(f'{path}: missing or external image in {camera}')
            images[out] = files
    return Episode(str(path), task, states, actions, images, resolve_fps(dataset, [meta], str(path)))
