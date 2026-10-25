"""Schema-checked ROS1 MCAP input; compressed images are decoded only when written."""
import json
from pathlib import Path
import numpy as np
from conversion.common.episode import Episode
from conversion.common.fps import recorded_metadata, resolve_fps
from conversion.common.task import resolve_task
from conversion.common.synchronization import select_captures

ROOT = Path(__file__).resolve().parents[2]


def schema_for(spec, path=None):
    schema = json.loads(Path(path or ROOT/'configs'/spec.name/'conversion.json').read_text())
    names = schema.get('joint_names', [])
    topics = [*schema.get('robot_topics', {}).values(), *schema.get('camera_topics', {}).values()]
    if (schema.get('robot') != spec.name or schema.get('fps') != 20
            or set(schema.get('robot_topics', {})) != set(spec.roles)
            or set(schema.get('camera_topics', {})) != set(spec.cameras)
            or schema.get('units') != {'joints': 'rad', 'gripper': 'm'}
            or len(names) != 7 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 7
            or any(not isinstance(t, str) or not t for t in topics) or len(set(topics)) != len(topics)):
        raise ValueError('Invalid recorder schema: require known roles, seven joint names, 20Hz, rad/m and unique topics')
    if spec.name == 'bimanual' and schema.get('camera_mapping') != spec.cameras:
        raise ValueError('Bimanual cameras must map wrist1=left_wrist and wrist2=right_wrist')
    return schema


def source_files(inputs):
    files = []
    for source in inputs:
        path = Path(source).resolve()
        if path.is_file() and path.suffix.lower() == '.mcap':
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p.resolve() for p in path.rglob('*.mcap')))
        else:
            raise ValueError(f'MCAP input not found: {path}')
    if not files or len(set(files)) != len(files):
        raise ValueError('No MCAP files or duplicate MCAP inputs')
    return files


def read_streams(path, schema):
    from mcap.reader import make_reader
    from mcap_ros1.decoder import DecoderFactory
    topics = {topic: role for role, topic in {**schema['robot_topics'], **schema['camera_topics']}.items()}
    streams = {role: [] for role in topics.values()}
    with Path(path).open('rb') as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for ros_type, channel, _, message in reader.iter_decoded_messages(topics=list(topics)):
            role = topics[channel.topic]
            robot = role in schema['robot_topics']
            expected = 'sensor_msgs/JointState' if robot else 'sensor_msgs/CompressedImage'
            if ros_type is None or ros_type.name != expected:
                raise ValueError(f'{path}: {channel.topic}: expected ROS type {expected}')
            stamp = message.header.stamp
            ns = stamp.secs*1_000_000_000 + stamp.nsecs
            if ns <= 0 or not 0 <= stamp.nsecs < 1_000_000_000:
                raise ValueError(f'{path}: {role}: invalid header timestamp')
            value = np.asarray(message.position, np.float32) if robot else bytes(message.data)
            if robot and (list(message.name) != schema['joint_names'] or value.shape != (7,) or not np.isfinite(value).all()):
                raise ValueError(f'{path}: {role}: invalid joint names, dimension or finite values')
            if not robot and not value:
                raise ValueError(f'{path}: {role}: empty compressed image')
            streams[role].append((ns, value))
    for role, values in streams.items():
        if not values:
            raise ValueError(f'{path}: missing MCAP stream {role}')
        values.sort(key=lambda v: v[0])
    return streams


def read_episode(path, spec, schema, task, timings):
    dataset, metadata = recorded_metadata(path)
    task = resolve_task(task, metadata, dataset, str(path))
    fps = resolve_fps(dataset, [metadata], str(path))
    with timings.measure('mcap_read'):
        streams = read_streams(path, schema)
    with timings.measure('synchronization'):
        stamps = {r: np.asarray([t for t, _ in values], np.int64) for r, values in streams.items()}
        _, selected = select_captures(stamps, spec, fps)
        def vectors(kind):
            return np.concatenate([np.stack([streams[r][i][1] for i in selected[r]])
                                   for r in spec.roles if r.endswith(kind)], axis=1)
        images = {out: [streams[c][i][1] for i in selected[c]] for c, out in spec.cameras.items()}
        return Episode(str(path), task, vectors('state'), vectors('action'), images, fps)
