"""Decode the inspected single-PiPER ROS1 recorder to the existing raw format.

Topics/names were checked against 260826_50epi/20260825_224317.mcap.
Only strict synchronization is supported: no synthetic images or joint values.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import hashlib
import uuid
import numpy as np
from PIL import Image

from conversion.common.fps import recorded_metadata, resolve_fps, consistent_fps, period_ns
from conversion.common.task import resolve_task
from conversion.single.mcap_io import schema_for, source_files, read_streams, sha256


def extract(inputs, destination, task, schema_path=None):
    schema = schema_for("single", schema_path)
    paths = source_files(inputs)
    recorded = [recorded_metadata(path) for path in paths]
    fps = consistent_fps(resolve_fps(dataset, [meta], str(path)) for path, (dataset, meta) in zip(paths, recorded))
    tasks = [resolve_task(task, meta, dataset, str(path)) for path, (dataset, meta) in zip(paths, recorded)]
    if destination.exists():
        raise FileExistsError(f'Extracted output exists: {destination}')
    final_destination = destination
    destination = destination.with_name(f'.{destination.name}.partial-{uuid.uuid4().hex[:8]}')
    destination.mkdir(parents=True)
    metadata = dict(format='piper_raw_fixed_rate_synced_v5', fps=fps, configured_fps=fps,
                    state_dim=7, value_names=[f'joint_{i}' for i in range(1,7)]+['gripper'],
                    robot_type='piper_single_arm', task=tasks[0],
                    observation_state_source=schema['robot_topics']['state'], action_source=schema['robot_topics']['action'])
    (destination / 'dataset_meta.json').write_text(json.dumps(metadata, indent=2))
    for index, path in enumerate(paths):
        streams, checksum = read_streams(path, schema)
        stamps = {k:np.asarray([t for t,_ in stream], dtype=np.int64) for k,stream in streams.items()}
        start = max(int(v[0]) for v in stamps.values())
        end = min(int(v[-1]) for v in stamps.values())
        grid = np.arange(start, end+1, period_ns(fps), dtype=np.int64)
        if len(grid) < 2:
            raise ValueError('Insufficient overlapping stream duration')
        episode = destination / f'episode_{index:06d}'
        episode.mkdir()
        chosen = {}
        for role, times in stamps.items():
            pos = np.searchsorted(times, grid)
            lo = np.maximum(pos-1, 0)
            hi = np.minimum(pos, len(times)-1)
            indices = np.where(abs(times[lo]-grid) <= abs(times[hi]-grid), lo, hi)
            tolerance = 30_000_000 if role in ('state','action') else 100_000_000
            if np.any(abs(times[indices]-grid) > tolerance):
                raise ValueError(f'{path}: {role} exceeds synchronization tolerance; inspect source. Partial extraction retained: {destination}')
            chosen[role] = indices
        if np.any(abs(stamps['state'][chosen['state']] - stamps['action'][chosen['action']]) > 30_000_000):
            raise ValueError('State/action capture times differ by more than 30ms')
        for camera in ('front','wrist'):
            (episode / camera).mkdir()
            for row, source_index in enumerate(chosen[camera]):
                with Image.open(io.BytesIO(streams[camera][source_index][1])) as image:
                    image.convert('RGB').save(episode / camera / f'{row:06d}.jpg', quality=95)
        np.savez(episode / 'data.npz',
                 observation_state=np.stack([streams['state'][i][1] for i in chosen['state']]),
                 action=np.stack([streams['action'][i][1] for i in chosen['action']]),
                 timestamp=np.arange(len(grid), dtype=np.float64)/fps,
                 unix_timestamp=grid/1e9)
        if sha256(path) != checksum:
            raise ValueError(f'MCAP source changed during extraction: {path}')
        (episode / 'meta.json').write_text(json.dumps(dict(metadata, task=tasks[index], successful=True, num_frames=len(grid),
             source_mcap=str(path), source_sha256=checksum, mcap_schema=schema, synchronization='nearest header capture, robot 30ms, image 100ms'), indent=2))
    destination.rename(final_destination)
    return final_destination
