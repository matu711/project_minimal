"""Basic array/Parquet/video checks; no full-frame image reader round trip."""
import json
import subprocess
import numpy as np


def check_episode(episode, spec):
    n = len(episode.states)
    for name, array in [('State', episode.states), ('Action', episode.actions)]:
        if array.shape != (n, spec.dim) or n < 1 or not np.isfinite(array).all():
            raise ValueError(f'{episode.source}: {name} must be finite (N,{spec.dim}), N >= 1')
    if set(episode.images) != set(spec.cameras.values()) or any(len(v) != n for v in episode.images.values()):
        raise ValueError(f'{episode.source}: missing camera or image/frame count mismatch')


def verify(root, spec, summaries):
    import pyarrow.parquet as pq
    info = json.loads((root/'meta/info.json').read_text())
    episodes = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines()]
    tasks = [json.loads(line) for line in (root/'meta/tasks.jsonl').read_text().splitlines()]
    expected = sum(ep['frames'] for ep in summaries)
    if (info['codebase_version'] != 'v2.1' or info['fps'] != summaries[0]['fps'] or info['robot_type'] != spec.robot_type
            or len(episodes) != len(summaries) or info['total_episodes'] != len(episodes)
            or info['total_frames'] != expected):
        raise ValueError('LeRobot metadata/FPS/episode/frame count mismatch')
    if len(list((root/'data').rglob('*.parquet'))) != len(episodes) or len(list((root/'videos').rglob('*.mp4'))) != len(episodes)*len(spec.cameras):
        raise ValueError('LeRobot Parquet/video file count mismatch')
    for key in ('observation.state', 'action'):
        feature = info['features'][key]
        if feature['shape'] != [spec.dim] or feature['dtype'] != 'float32' or feature['names'] != list(spec.value_names):
            raise ValueError(f'LeRobot feature mismatch: {key}')
    for ep, summary in zip(episodes, summaries, strict=True):
        variables = dict(episode_index=ep['episode_index'], episode_chunk=ep['episode_index']//info['chunks_size'])
        table = pq.read_table(root/info['data_path'].format(**variables)).to_pydict()
        n = summary['frames']
        if ep['length'] != n or ep['tasks'] != [summary['task']]:
            raise ValueError('LeRobot episode length/task mismatch')
        for key in ('observation.state', 'action'):
            values = np.asarray(table[key])
            if values.shape != (n, spec.dim) or not np.isfinite(values).all():
                raise ValueError(f'Invalid output {key}')
        np.testing.assert_allclose(table['timestamp'], np.arange(n)/summary['fps'], atol=1e-4, rtol=0)
        if any(tasks[int(i)]['task'] != summary['task'] for i in table['task_index']):
            raise ValueError('LeRobot frame task mismatch')
        if spec.name == 'bimanual' and not np.asarray(table['quality.frame_valid'], bool).all():
            raise ValueError('Strict conversion must contain only valid frames')
        for camera, shape in summary['shapes'].items():
            key = f'observation.images.{camera}'
            if info['features'][key]['shape'] != list(shape):
                raise ValueError(f'LeRobot camera shape mismatch: {camera}')
            video = root/info['video_path'].format(**variables, video_key=key)
            result = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames','-of','json',str(video)],
                capture_output=True, text=True, check=True)
            stream = json.loads(result.stdout)['streams'][0]
            a, b = map(int, stream['avg_frame_rate'].split('/'))
            if int(stream['nb_read_frames']) != n or a/b != summary['fps'] or (stream['height'],stream['width'],3) != tuple(shape):
                raise ValueError(f'Video frame count/FPS/shape mismatch: {video}')
    return dict(episodes=len(episodes), frames=expected, parquet_checked=True, video_frame_counts_checked=True)
