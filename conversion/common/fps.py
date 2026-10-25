"""Resolve nominal output FPS without confusing it with measured episode rate."""
import json
import warnings
from pathlib import Path


def positive_fps(value):
    try:
        fps = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'Invalid metadata FPS: {value!r}') from error
    if fps <= 0:
        raise ValueError(f'Metadata FPS must be positive: {value!r}')
    return fps


def resolve_fps(dataset, episodes=(), label='input'):
    if dataset.get('fps') is not None:
        return positive_fps(dataset['fps'])
    values = {positive_fps(ep['fps']) for ep in episodes if ep.get('fps') is not None}
    if len(values) > 1:
        raise ValueError(f'{label}: conflicting episode FPS: {sorted(values)}')
    if values:
        return values.pop()
    warnings.warn(f'{label}: no metadata FPS; using nominal 20 FPS', RuntimeWarning)
    return 20


def consistent_fps(values):
    values = set(values)
    if len(values) != 1:
        raise ValueError(f'Conflicting input FPS: {sorted(values)}')
    return values.pop()


def period_ns(fps):
    period = round(1_000_000_000 / positive_fps(fps))
    if period < 1:
        raise ValueError('FPS exceeds nanosecond clock resolution')
    return period


def recorded_metadata(path):
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    episode_path = directory/'meta.json'
    dataset_path = directory/'dataset_meta.json'
    if not dataset_path.is_file():
        dataset_path = directory.parent/'dataset_meta.json'
    read = lambda p: json.loads(p.read_text()) if p.is_file() else {}
    return read(dataset_path), read(episode_path)
