"""Small conversion records; actions remain absolute joint/gripper targets."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class RobotSpec:
    name: str
    robot_type: str
    arms: tuple
    cameras: dict
    value_names: tuple
    grid_uses_cameras: bool = False  # Legacy adapter field; selection now always intersects all streams.

    @property
    def roles(self):
        return tuple(f'{arm}_{kind}' if arm else kind for arm in self.arms for kind in ('state', 'action'))

    @property
    def dim(self):
        return 7 * len(self.arms)


@dataclass
class Episode:
    source: str
    task: str
    states: np.ndarray
    actions: np.ndarray
    images: dict
    fps: int = 20


@dataclass
class Timings:
    values: dict = field(default_factory=dict)

    @contextmanager
    def measure(self, stage):
        start = perf_counter()
        try:
            yield
        finally:
            self.values[stage] = self.values.get(stage, 0.) + perf_counter() - start
