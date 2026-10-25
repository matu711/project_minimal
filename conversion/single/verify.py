"""Verify completed single-arm LeRobot files before publishing a final directory."""
import json
import subprocess
import numpy as np


def verify(root):
    import pyarrow.parquet as pq
    info = json.loads((root / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v2.1' or info['robot_type'] != 'piper_single_arm':
        raise ValueError('Expected single-PiPER LeRobot v2.1')
    for key in ('action','observation.state'):
        if info['features'][key]['shape'] != [7]:
            raise ValueError('Expected 7D joint/gripper values')
    episodes = [json.loads(s) for s in (root/'meta/episodes.jsonl').read_text().splitlines() if s]
    frames = 0
    for ep in episodes:
        idx = ep['episode_index']
        variables = dict(episode_index=idx, episode_chunk=idx//info['chunks_size'])
        table = pq.read_table(root / info['data_path'].format(**variables))
        if len(table) != ep['length']:
            raise ValueError('Parquet episode length differs from metadata')
        for key in ('action','observation.state'):
            values = np.asarray(table[key].to_pylist())
            if values.shape != (len(table),7) or not np.isfinite(values).all():
                raise ValueError(f'Invalid numeric values: {key}')
        times = np.asarray(table['timestamp'].to_pylist())
        np.testing.assert_allclose(times, np.arange(len(times))/info['fps'], atol=1e-4, rtol=0)
        for camera in ('front','wrist'):
            path = root / info['video_path'].format(**variables, video_key=f'observation.images.{camera}')
            result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=nb_read_frames,width,height,avg_frame_rate','-of','json',str(path)],
                capture_output=True,text=True,check=True)
            stream = json.loads(result.stdout)['streams'][0]
            numerator, denominator = map(int,stream['avg_frame_rate'].split('/'))
            shape = info['features'][f'observation.images.{camera}']['shape']
            if int(stream['nb_read_frames']) != len(table) or numerator/denominator != info['fps'] or [int(stream['height']),int(stream['width']),3] != shape:
                raise ValueError('Video count, dimensions or fps mismatch')
        frames += len(table)
    if len(episodes) != info['total_episodes'] or frames != info['total_frames'] or frames == 0:
        raise ValueError('Dataset total counts mismatch')
    return dict(episodes=len(episodes),frames=frames,parquet_checked=True,video_frame_counts_checked=True)
