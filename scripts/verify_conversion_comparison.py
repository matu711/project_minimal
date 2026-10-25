#!/usr/bin/env python3
"""Compare benchmark outputs and exercise unchanged pi0/pi05 training transforms."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), '/opt/project_minimal/openpi', '/opt/project_minimal/openpi/src',
                '/opt/project_minimal/openpi/packages/openpi-client/src']


def table(root):
    import pyarrow.parquet as pq
    info = json.loads((root/'meta/info.json').read_text())
    episodes = [json.loads(s) for s in (root/'meta/episodes.jsonl').read_text().splitlines()]
    rows = []
    for ep in episodes:
        data = dict(episode_index=ep['episode_index'],episode_chunk=ep['episode_index']//info['chunks_size'])
        rows.append(pq.read_table(root/info['data_path'].format(**data)).to_pydict())
    return info, episodes, rows


def main():
    from PIL import Image
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from conversion.pipeline import robot_spec
    from training import train
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args()
    folder = args.directory.resolve()
    results = []
    for robot, kind in [('single','raw'),('single','mcap'),('bimanual','raw'),('bimanual','excerpt')]:
        label = f'{robot}_{kind}'
        before, after = folder/f'before_{label}', folder/f'after_{label}'
        bi, be, bt = table(before)
        ai, ae, at = table(after)
        assert (bi['total_frames'],len(be)) == (ai['total_frames'],len(ae))
        spec = robot_spec(robot)
        for old,new,old_ep,new_ep in zip(bt,at,be,ae,strict=True):
            for key in ['observation.state','action','task_index','episode_index','frame_index']:
                np.testing.assert_array_equal(new[key],old[key])
            np.testing.assert_allclose(new['timestamp'],old['timestamp'],atol=1e-4,rtol=0)
            assert old_ep['tasks'] == new_ep['tasks']
        assert (before/'meta/tasks.jsonl').read_text() == (after/'meta/tasks.jsonl').read_text()
        # Sampled video comparison is a regression check, not part of conversion.
        old_ds = LeRobotDataset(repo_id='local/benchmark',root=before,video_backend='pyav')
        new_ds = LeRobotDataset(repo_id='local/benchmark',root=after,video_backend='pyav')
        image_errors = {}
        for camera in spec.cameras.values():
            key = f'observation.images.{camera}'
            errors = []
            for row in [0,ai['total_frames']//2,ai['total_frames']-1]:
                old = np.moveaxis(old_ds[row][key].numpy(),0,-1)
                new = np.moveaxis(new_ds[row][key].numpy(),0,-1)
                assert old.shape == new.shape
                old = np.asarray(Image.fromarray((old*255).round().astype(np.uint8)).resize((32,32)),float)
                new = np.asarray(Image.fromarray((new*255).round().astype(np.uint8)).resize((32,32)),float)
                errors.append(float(abs(old-new).mean()))
            assert max(errors) < 12, (label,camera,errors)
            image_errors[camera] = errors
        # Check every selected input image reference without adding this to production.
        if kind in ('mcap','excerpt'):
            import importlib
            from conversion.common import mcap_io
            from conversion.common.episode import Timings
            schema = mcap_io.schema_for(spec)
            old_timing=json.loads((folder/f'before_{label}.timing.json').read_text())
            source=Path(old_timing['input'])
            episode=mcap_io.read_episode(source,spec,schema,be[0]['tasks'][0],Timings())
            if robot=='single':
                old_images=folder/'before_single_mcap_raw/episode_000000'
                for camera,values in episode.images.items():
                    import io
                    for row,payload in enumerate(values):
                        generated=io.BytesIO()
                        with Image.open(io.BytesIO(payload)) as image:image.convert('RGB').save(generated,format='JPEG',quality=95)
                        assert generated.getvalue()==(old_images/camera/f'{row:06d}.jpg').read_bytes()
            else:
                manifest=json.loads((before/'provenance/episode_000000.json').read_text())
                # MCAP source files are indexed in header order; reconstruct IDs below.
                streams=mcap_io.read_streams(source,schema)
                for camera,out in spec.cameras.items():
                    payloads=[payload for _,payload in streams[camera]]
                    for frame,payload in zip(manifest['frames'],episode.images[out],strict=True):
                        index=int(Path(frame['images'][camera]).stem)
                        assert payloads[index]==payload
        if robot == 'bimanual' and kind == 'raw':
            from conversion.common import raw_reader
            from conversion.common.episode import Timings
            old_timing=json.loads((folder/f'before_{label}.timing.json').read_text())
            source=Path(old_timing['input'])
            path,dataset,meta=raw_reader.raw_episodes([source],spec)[0]
            episode=raw_reader.read_episode(path,dataset,meta,spec,None,Timings())
            manifest=json.loads((before/'provenance/episode_000000.json').read_text())
            for camera,out in spec.cameras.items():
                for frame,new_image in zip(manifest['frames'],episode.images[out],strict=True):
                    assert (path/frame['images'][camera]).read_bytes()==new_image.read_bytes()
        training=[]
        import os
        from lerobot.common.datasets import lerobot_dataset
        for model in ['pi0','pi05']:
            params=train.load_params(ROOT/'configs'/robot/f'{model}_lora.yaml')
            # Use the output directly under an isolated LeRobot repo root.
            repo='local/'+after.name
            data_root=folder/'training_data'
            (data_root/'local').mkdir(parents=True,exist_ok=True)
            link=data_root/repo
            if not link.exists():link.symlink_to(after,target_is_directory=True)
            params.update(data_dir=str(data_root),batch_size=1,num_workers=0,action_horizon=20 if robot=='single' else 50,
                          assets_dir=str(folder/'training_assets'))
            params['data_collection']['dataset_repo_id']=repo
            cfg=train.build_config(params)
            lerobot_dataset.HF_LEROBOT_HOME=data_root
            check=train.check_dataset(cfg,robot)
            batch=train.validate_batch(cfg,robot)
            training.append(dict(model=model,check=check,batch=batch))
        results.append(dict(case=label,frames=ai['total_frames'],episodes=len(ae),
                            state_action_exact=True,task_exact=True,camera_sample_errors=image_errors,training=training))
    (folder/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
