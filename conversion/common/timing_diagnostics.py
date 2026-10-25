"""Observe raw clocks without changing recorded-row acceptance or output."""
import json
from pathlib import Path
import uuid
import warnings
import numpy as np


def _numbers(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def record_timing(episodes, fps, output, destination=None, robot="single"):
    destination = Path(destination) if destination else Path(output).with_name(
        Path(output).name + '.raw_timing-' + uuid.uuid4().hex[:8] + '.json')
    report = dict(version=1, output_clock='nominal frame_index / resolved_fps; NOT measured capture time',
                  fps=fps, log_path=str(destination.resolve()), episodes=[])
    nominal_period = 1/fps if fps > 0 else None
    for path in episodes:
        entry = dict(source=str(path.resolve()), clocks={}, warnings=[])
        with np.load(path/'data.npz', allow_pickle=False) as data:
            count = len(data['observation_state'])
            entry['raw_frames'] = count
            for key in data.files:
                if 'timestamp' not in key:
                    continue
                try:
                    raw = data[key]
                    if raw.shape != (count,) or not (np.issubdtype(raw.dtype, np.floating) or np.issubdtype(raw.dtype, np.integer)):
                        entry['warnings'].append(f'{key}: clock shape/type cannot be audited; legacy acceptance retained')
                        continue
                    values = np.asarray(raw, np.float64)
                    deltas = np.diff(values)
                    # Only frame-grid clocks are expected to advance at dataset FPS.
                    irregular = np.flatnonzero((~np.isfinite(deltas) | (abs(deltas-nominal_period)>1e-4)) if nominal_period is not None else np.ones(len(deltas),dtype=bool))
                    clock = dict(timestamps_s=_numbers(values), actual_intervals_s=_numbers(deltas))
                    if key in ('timestamp','unix_timestamp'):
                        clock['irregular_intervals'] = [dict(row_from=int(i),row_to=int(i+1),
                            actual_delta_s=float(deltas[i]) if np.isfinite(deltas[i]) else None,
                            nominal_delta_s=nominal_period) for i in irregular]
                        if len(irregular) or not np.isfinite(values).all():
                            actual = _numbers(deltas[irregular[:5]])
                            entry['warnings'].append(f'{key}: {len(irregular)} irregular intervals; first actual dt(s)={actual}, nominal={nominal_period}; output keeps nominal timestamps and recorded sample order')
                    entry['clocks'][key] = clock
                except (ValueError, TypeError) as error:
                    entry['warnings'].append(f'{key}: unavailable clock ({error}); legacy acceptance retained')
            if not any(k in entry['clocks'] for k in ('timestamp','unix_timestamp')):
                entry['warnings'].append('No usable measured frame clock; output timestamps are nominal only')
        for message in entry['warnings']:
            warnings.warn(f'{robot} raw timing: {path.name}: {message}. Actual clock log: {destination}', RuntimeWarning)
        report['episodes'].append(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Raw timing log: {destination} (output timestamps are nominal, not measured)')
    return report
