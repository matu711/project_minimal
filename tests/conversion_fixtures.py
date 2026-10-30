"""Small real-writer fixtures for both robot layouts."""
import csv
import json
import numpy as np
from PIL import Image
from conversion.pipeline import robot_spec


def record(root, robot='bimanual', count=4):
    spec = robot_spec(robot)
    path = root/'episode_000000'
    path.mkdir(parents=True)
    times = np.arange(count, dtype=np.float64)/20
    unix = 1_792_033_200 + times
    states = np.arange(count*spec.dim, dtype=np.float32).reshape(count,spec.dim)/100
    actions = states + .1
    values = dict(observation_state=states, action=actions, timestamp=times, unix_timestamp=unix,
                  **{r+'_ros_timestamp': unix for r in spec.roles},
                  **{c+'_ros_timestamp': unix for c in spec.cameras})
    np.savez(path/'data.npz', **values)
    metadata=dict(fps=20, state_dim=spec.dim, task='  Stack cubes.\nKeep order.  ')
    if robot == 'single':
        metadata.update(format='piper_raw_fixed_rate_synced_v5', value_names=list(spec.value_names), robot_type=spec.robot_type)
    (root/'dataset_meta.json').write_text(json.dumps(metadata))
    (path/'meta.json').write_text(json.dumps(dict(task='  Stack cubes.\nKeep order.  ', num_frames=count, completed=True)))
    with (path/'frames.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['frame_index','timestamp',*(c+'_image' for c in spec.cameras)])
        writer.writeheader()
        for row in range(count):
            entry = dict(frame_index=row,timestamp=times[row])
            for i,camera in enumerate(spec.cameras):
                (path/camera).mkdir(exist_ok=True)
                name = f'{camera}/{row:06d}.jpg' if robot == 'single' else f'{camera}/{count-row:06d}.png'
                Image.new('RGB',(64,48) if robot == 'single' else (64+16*i,48+16*i),(30+i*70,row*20,60)).save(path/name)
                entry[camera+'_image'] = name
            writer.writerow(entry)
    return path
