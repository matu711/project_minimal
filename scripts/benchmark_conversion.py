#!/usr/bin/env python3
"""Measure old/new conversion stages in-process, using the same LeRobot environment."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import functools
import importlib
import json
from pathlib import Path
import sys
import time


class Clock:
    def __init__(self):
        self.elapsed = defaultdict(float)
        self.active = [(0, 'other')]
        self.last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.elapsed[max(self.active)[1]] += now - self.last
        self.last = now

    @contextmanager
    def stage(self, name, priority=10):
        self.tick()
        item = (priority, name)
        self.active.append(item)
        try:
            yield
        finally:
            self.tick()
            self.active.remove(item)

    def patch(self, obj, name, category, priority):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.stage(category, priority):
                return original(*args, **kwargs)
        setattr(obj, name, wrapped)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', choices=['before', 'after'], required=True)
    p.add_argument('--robot', choices=['single', 'bimanual'], required=True)
    p.add_argument('--kind', choices=['mcap', 'raw'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='pick up the white object')
    a = p.parse_args()
    sys.path.insert(0, str(a.source_root))
    # Preload shared libraries so import startup is excluded equally from both versions.
    from PIL import Image
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet
    from mcap.reader import make_reader
    clock = Clock()
    clock.patch(Image, 'open', 'image_decode', 60)
    clock.patch(Image.Image, 'load', 'image_decode', 60)
    clock.patch(Image.Image, 'convert', 'image_decode', 60)
    clock.patch(LeRobotDataset, 'add_frame', 'write_encode', 40)
    clock.patch(LeRobotDataset, 'save_episode', 'write_encode', 40)
    started = time.perf_counter()
    clock.last = started
    report = dict(version=a.version, robot=a.robot, kind=a.kind, input=str(a.input), output=str(a.output))
    try:
        if a.version == 'after':
            from conversion.pipeline import convert
            from conversion.common import mcap_io, raw_reader, lerobot_writer, validation
            clock.patch(mcap_io, 'read_streams', 'mcap_read', 50)
            clock.patch(mcap_io, 'select_captures', 'synchronization', 30)
            clock.patch(raw_reader, 'read_episode', 'raw_read', 30)
            clock.patch(raw_reader, 'nearest_indices', 'synchronization', 50)
            clock.patch(raw_reader, 'check_raw_grid', 'synchronization', 50)
            clock.patch(lerobot_writer, 'write', 'write_encode', 20)
            clock.patch(validation, 'verify', 'verification', 100)
            result = convert(a.robot, [a.input], a.output, 'local/benchmark', task=a.task if a.kind == 'mcap' else None)
            report['pipeline_timings'] = result['timings_s']
        else:
            common = importlib.import_module('conversion.mcap_io')
            clock.patch(common, 'read_streams', 'mcap_read', 50)
            if a.robot == 'single':
                extractor = importlib.import_module('conversion.single.extract')
                writer = importlib.import_module('conversion.single.convert')
                verifier = importlib.import_module('conversion.single.verify')
                clock.patch(extractor, 'read_streams', 'mcap_read', 50)
                source = a.input
                if a.kind == 'mcap':
                    with clock.stage('synchronization', 20):
                        source = extractor.extract([source], a.output.with_name(a.output.name+'_raw'), a.task)
                with clock.stage('raw_read', 10):
                    sys.argv = ['convert', '--input', str(source), '--output', str(a.output), '--repo-id', 'local/benchmark']
                    writer.main()
                with clock.stage('verification', 100):
                    verifier.verify(a.output)
            else:
                writer = importlib.import_module('conversion.bimanual.convert')
                recovery = importlib.import_module('conversion.bimanual.recovery')
                clock.patch(writer, 'verify', 'verification', 100)
                clock.patch(writer, 'inspect_episode', 'raw_read', 10)
                for name in ['_robot_group','_nearest','synchronize_saved_episode']:
                    clock.patch(recovery, name, 'synchronization', 30)
                clock.patch(recovery, 'recover_episode', 'mcap_read', 20)
                if a.kind == 'mcap':
                    extractor = importlib.import_module('conversion.bimanual.extract')
                    clock.patch(extractor, 'read_streams', 'mcap_read', 50)
                    with clock.stage('synchronization', 20):
                        extractor.main(['--input', str(a.input), '--output', str(a.output), '--repo-id','local/benchmark','--task',a.task])
                else:
                    writer.main(['--input',str(a.input),'--output',str(a.output),'--repo-id','local/benchmark','--gap-policy','recover','--image-policy','flag'])
        info = json.loads((a.output/'meta/info.json').read_text())
        report.update(status='ok', frames=info['total_frames'], episodes=info['total_episodes'])
    except Exception as error:
        report.update(status='error', error=str(error))
        raise
    finally:
        clock.tick()
        report.update(total_s=time.perf_counter()-started, stages_s=dict(clock.elapsed))
        if report.get('frames'):
            report['ms_per_frame'] = report['total_s']*1000/report['frames']
        a.output.with_suffix('.timing.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
