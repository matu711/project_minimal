"""Legacy single-arm MCAP checks on top of the shared ROS1 decoder."""
import hashlib
import json
import io
from pathlib import Path
from PIL import Image
from conversion.common import mcap_io
ROOT = Path(__file__).resolve().parents[2]

def source_files(inputs):
    paths = []
    for path in inputs:
        path = Path(path).resolve()
        if path.is_file() and path.suffix.lower() == '.mcap':
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p.resolve() for p in path.rglob('*.mcap')))
        else:
            raise FileNotFoundError(f'MCAP input not found: {path}')
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('No MCAP files or duplicate MCAP inputs')
    return paths


def schema_for(robot, path=None):
    schema = json.loads(Path(path or ROOT / 'configs' / robot / 'conversion.json').read_text())
    names = schema.get('joint_names')
    if (not isinstance(names, list) or len(names) != 7
            or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 7):
        raise ValueError('MCAP schema must define seven unique joint/gripper names in recorded order')
    expected_robot = {'state', 'action'} if robot == 'single' else {
        'left_state', 'left_action', 'right_state', 'right_action'}
    expected_cameras = {'front', 'wrist'} if robot == 'single' else {'front', 'wrist1', 'wrist2'}
    if (schema.get('robot') != robot or schema.get('fps') != 20
            or set(schema.get('robot_topics', {})) != expected_robot
            or set(schema.get('camera_topics', {})) != expected_cameras
            or schema.get('units') != {'joints': 'rad', 'gripper': 'm'}
            or len(schema.get('joint_names', [])) != 7):
        raise ValueError('MCAP schema must describe the inspected 20Hz joint-rad/gripper-m recorder')
    topics = [*schema['robot_topics'].values(), *schema['camera_topics'].values()]
    if any(not isinstance(topic, str) or not topic for topic in topics) or len(set(topics)) != len(topics):
        raise ValueError('MCAP topic names must be nonempty and unique')
    if robot == 'bimanual' and schema.get('camera_mapping') != {
            'front': 'front', 'wrist1': 'left_wrist', 'wrist2': 'right_wrist'}:
        raise ValueError('Canonical dual-PiPER camera mapping must be confirmed before changing it')
    return schema

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_streams(path, schema):
    checksum = sha256(path)
    streams = mcap_io.read_streams(path, schema)
    for camera in schema['camera_topics']:
        for _, payload in streams[camera]:
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
    if sha256(path) != checksum:
        raise ValueError(f'MCAP source changed during decoding: {path}')
    return streams, checksum
