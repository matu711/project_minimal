"""Inspected single-PiPER topic roles and output layout."""
from conversion.common.episode import RobotSpec

SPEC = RobotSpec('single', 'piper_single_arm', ('',),
                 {'front': 'front', 'wrist': 'wrist'},
                 tuple([f'joint_{i}' for i in range(1, 7)] + ['gripper']),
                 grid_uses_cameras=True)
