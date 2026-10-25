"""Metadata-FPS nearest capture selection. Never interpolate or restore missing raw rows."""
import numpy as np
from conversion.common.fps import period_ns

ROBOT_NS = 30_000_000
CAMERA_NS = 100_000_000


def nearest_indices(stamps, targets, tolerance, label):
    stamps = np.asarray(stamps, dtype=np.int64)
    if stamps.ndim != 1 or not len(stamps) or np.any(np.diff(stamps) < 0):
        raise ValueError(f'{label}: missing or unordered capture timestamps')
    pos = np.searchsorted(stamps, targets)
    lo, hi = np.maximum(pos - 1, 0), np.minimum(pos, len(stamps) - 1)
    chosen = np.where(abs(stamps[lo] - targets) <= abs(stamps[hi] - targets), lo, hi)
    errors = abs(stamps[chosen] - targets)
    bad = np.flatnonzero(errors > tolerance)
    if len(bad):
        row = int(bad[0])
        raise ValueError(f'{label}: row {row}, nearest capture error {errors[row]/1e6:.3f}ms exceeds {tolerance/1e6:g}ms')
    return chosen


def select_captures(stamps, spec, fps=20):
    grid_roles = (*spec.roles, *spec.cameras)
    start = max(int(stamps[role][0]) for role in grid_roles)
    end = min(int(stamps[role][-1]) for role in grid_roles)
    grid = np.arange(start, end + 1, period_ns(fps), dtype=np.int64)
    if len(grid) < 2:
        raise ValueError('Insufficient overlapping capture duration for an episode')
    selected = {role: nearest_indices(times, grid, ROBOT_NS if role in spec.roles else CAMERA_NS, role)
                for role, times in stamps.items()}
    robot_times = np.stack([stamps[r][selected[r]] for r in spec.roles])
    bad = np.flatnonzero(np.ptp(robot_times, axis=0) > ROBOT_NS)
    if len(bad):
        raise ValueError(f'Robot synchronization: row {bad[0]}, State/Action capture span exceeds 30ms')
    return grid, selected
