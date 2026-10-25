"""Inspected dual-PiPER output: left seven values, then right seven."""
from conversion.common.episode import RobotSpec

SPEC = RobotSpec('bimanual', 'piper_bimanual', ('left', 'right'),
                 {'front': 'front', 'wrist1': 'left_wrist', 'wrist2': 'right_wrist'},
                 tuple(name for arm in ('left', 'right') for name in
                       [*(f'{arm}_joint_{i}_rad' for i in range(1, 7)), f'{arm}_gripper_m']))
