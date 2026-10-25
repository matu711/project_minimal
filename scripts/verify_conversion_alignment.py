#!/usr/bin/env python3
"""Read-only real-data regression; writes only to a fresh verification directory."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT),'/opt/project_minimal/openpi','/opt/project_minimal/openpi/src','/opt/project_minimal/openpi/packages/openpi-client/src']
from verify_single_compatibility import compare
from verify_conversion_comparison import table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'logs/single_compatibility/real')
    args=parser.parse_args()
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    from conversion.pipeline import convert,robot_spec
    from conversion.common import mcap_io,raw_reader
    from conversion.common.episode import Timings
    from conversion.common.synchronization import select_captures
    results=[]
    def save(): (folder/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    def run(label,robot,source,task=None,resize=None):
        start=time.perf_counter();print('BEGIN',label,flush=True)
        output=folder/label
        report=convert(robot,[source],output,'local/compat' if robot=='single' else 'local/alignment',task=task,resize=resize)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds=LeRobotDataset(repo_id='local/check',root=output,video_backend='pyav')
        info=json.loads((output/'meta/info.json').read_text())
        for row in [0,info['total_frames']//2,info['total_frames']-1]:
            item=ds[row]
            assert item['observation.state'].shape==(7 if robot=='single' else 14,)
            for camera in robot_spec(robot).cameras.values():
                assert np.isfinite(item[f'observation.images.{camera}'].numpy()).all()
        result=dict(case=label,report=report,elapsed_s=time.perf_counter()-start,reader_samples_passed=True)
        if robot=='single':
            result['baseline_comparison']=compare(args.baseline/('restored_'+('raw_450' if label=='single_raw' else 'mcap_503')),output)
        if robot=='bimanual' and label=='bimanual_raw':
            old_info,old_ep,old_data=table(ROOT/'logs/conversion_simplification/after_bimanual_raw')
            new_info,new_ep,new_data=table(output)
            for a,b in zip(old_data,new_data,strict=True):
                for key in ['observation.state','action','timestamp','frame_index','episode_index','task_index']:
                    np.testing.assert_array_equal(a[key],b[key])
            result['baseline_vectors_timestamps_indices_exact']=True
            result['baseline_full_comparison']=compare_dual_recorded_images(ROOT/'logs/conversion_simplification/after_bimanual_raw',output)
        results.append(result);save();print('PASS',label,report['frames'],flush=True)
        return output
    single=Path('/source/single/20260825_224317')
    run('single_raw','single',single)
    run('single_mcap','single',single/'20260825_224317.mcap',task='pick up the white object')
    dual=Path('/source/bimanual/20261015_164839')
    run('bimanual_raw','bimanual',dual)
    run('bimanual_irregular','bimanual',Path('/source/bimanual/20261015_165042'))
    run('bimanual_resize','bimanual',dual,task='Explicit recorded-data check.',resize=(224,224))
    # Compare actual header choices with the pre-change solver, including former endpoints.
    spec=robot_spec('bimanual');schema=mcap_io.schema_for(spec)
    text=subprocess.check_output(['git','show','f8e8c1e:conversion/common/synchronization.py'],cwd=ROOT,text=True)
    old={};exec(compile(text,'baseline_synchronization','exec'),old)
    for name,path in [('full',dual/'episode_1792050464343925529.mcap'),
                      ('excerpt',ROOT/'logs/conversion_simplification/inputs/bimanual_excerpt.mcap')]:
        streams=mcap_io.read_streams(path,schema)
        stamps={r:np.array([t for t,_ in stream],np.int64) for r,stream in streams.items()}
        entry=dict(case='mcap_selection_'+name)
        for version,selector in [('before',old['select_captures']),('after',select_captures)]:
            try:
                grid,selection=selector(stamps,spec)
                entry[version]=dict(frames=len(grid),start_ns=int(grid[0]),end_ns=int(grid[-1]))
                entry[version+'_arrays']=(grid,selection)
            except ValueError as error:entry[version]=dict(error=str(error))
        if 'before_arrays' in entry and 'after_arrays' in entry:
            bg,bs=entry['before_arrays'];ag,ass=entry['after_arrays']
            # Grid origin can shift when all cameras define the intersection.
            entry['same_grid_targets']=bool(np.array_equal(bg,ag))
            entry['common_exact_targets']=int(len(np.intersect1d(bg,ag)))
            ai,bi=np.nonzero(ag[:,None]==bg[None,:])
            entry['changed_selection_on_common_targets']={r:int(np.sum(ass[r][ai]!=bs[r][bi])) for r in spec.roles}
            entry['max_grid_origin_shift_ms']=float((ag[0]-bg[0])/1e6)
        for key in ['before_arrays','after_arrays']:entry.pop(key,None)
        results.append(entry);save()
        if 'error' not in entry['after']:
            run('bimanual_mcap_'+name,'bimanual',path,task='Assembling a box.')
        else:
            print('EXPECTED STRICT REJECTION',name,entry['after'],flush=True)
    # All 50 real raw episodes are input-validated without encoding them all.
    accepted=convert('single',[Path('/source/single')],folder/'all_single_validation','local/check',validate_only=True)
    results.append(dict(case='single_all_episodes',episodes=len(accepted['episodes']),frames=sum(e['frames'] for e in accepted['episodes'])));save()
    # Unmodified training code, real video/state/action batches, no model weights/training.
    from training import train
    from lerobot.common.datasets import lerobot_dataset
    for result in results:
        if 'report' not in result:continue
        label=result['case'];robot='single' if label.startswith('single') else 'bimanual'
        output=folder/label;checks=[]
        for model in ['pi0','pi05']:
            params=train.load_params(ROOT/'configs'/robot/f'{model}_lora.yaml')
            data_root=folder/'training_data';(data_root/'local').mkdir(parents=True,exist_ok=True)
            link=data_root/'local'/label
            if not link.exists():link.symlink_to(output,target_is_directory=True)
            params.update(data_dir=str(data_root),batch_size=1,num_workers=0,action_horizon=20 if robot=='single' else 50,
                          assets_dir=str(folder/'training_assets'))
            params['data_collection']['dataset_repo_id']='local/'+label
            cfg=train.build_config(params);lerobot_dataset.HF_LEROBOT_HOME=data_root
            checks.append(dict(model=model,check=train.check_dataset(cfg,robot),batch=train.validate_batch(cfg,robot)))
        result['training']=checks;save();print('TRAINING BATCH PASS',label,flush=True)
    # Public shell entrypoints, including formerly rejected irregular raw.
    cli=[]
    for robot,source in [('single',single),('single',single/'20260825_224317.mcap'),('bimanual',dual),('bimanual',Path('/source/bimanual/20261015_165042')),('bimanual',ROOT/'logs/conversion_simplification/inputs/bimanual_excerpt.mcap')]:
        label=robot+'_'+str(len(cli))
        command=['sh','scripts/convert.sh','--robot',robot,'--dataset',label,'--input',str(source),'--output',str(folder/('cli_'+label)),
                 '--validate-only','--resize','224','224','--task','CLI check.']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        cli.append(dict(command=command,returncode=result.returncode,stderr=result.stderr));assert result.returncode==0,(command,result.stderr)
    (folder/'cli.json').write_text(json.dumps(cli,indent=2)+'\n')
    compare_mcap_same_targets(ROOT/'logs/conversion_simplification/inputs/bimanual_excerpt.mcap',folder/'mcap-same-targets.json')
    check_ee_compatibility(folder/'ee',args.baseline)
    print('ALL DONE',flush=True)


def check_ee_compatibility(folder, baseline):
    """Use explicit recorded tasks equal to the former replacement result."""
    import shutil
    from conversion.pipeline import convert
    folder=Path(folder);baseline=Path(baseline)
    inputs=[]
    for label,task in [('ee_a','Pick B'),('ee_b','Pick D')]:
        source=folder/(label+'_inputs')
        shutil.copytree(baseline/label,source)
        for meta_path in source.glob('*/meta.json'):
            meta=json.loads(meta_path.read_text());meta['task']=task
            meta_path.write_text(json.dumps(meta))
        inputs.append(source)
    checks=[]
    for mode,name in [('joint_absolute','ee_absolute'),('ee_delta_axis_angle','ee_delta')]:
        output=folder/name
        convert('single',inputs,output,'local/compat',resize=(32,24),
                single_options=['--include-failed','--action-mode',mode])
        result=compare(baseline/('restored_'+name),output)
        result['case']=name;checks.append(result)
    (folder/'ee-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks


def compare_dual_recorded_images(before, after):
    """Raw row-preserving images intentionally differ from former re-selection."""
    import hashlib
    before,after=Path(before),Path(after)
    bi,be,bt=table(before);ai,ae,at=table(after)
    assert bi==ai and be==ae
    for old,new in zip(bt,at,strict=True):
        assert old.keys()==new.keys()
        for key in old:np.testing.assert_array_equal(old[key],new[key])
    stats_a=json.loads((before/'meta/episodes_stats.jsonl').read_text())['stats']
    stats_b=json.loads((after/'meta/episodes_stats.jsonl').read_text())['stats']
    changed=[key for key in stats_a if stats_a[key]!=stats_b[key]]
    assert all(key.startswith('observation.images.') for key in changed)
    videos={str(p.relative_to(before)):hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((after/p.relative_to(before)).read_bytes()).hexdigest()
            for p in sorted((before/'videos').rglob('*.mp4'))}
    return dict(all_parquet_fields_exact=True,info_and_episodes_exact=True,
                intentionally_changed_image_stats=changed,encoded_video_exact=videos)


def compare_mcap_same_targets(path, destination):
    """Use Git baseline solver on the new grid to isolate selection from grid changes."""
    from conversion.pipeline import robot_spec
    from conversion.common import mcap_io
    from conversion.common.synchronization import select_captures
    spec=robot_spec('bimanual')
    streams=mcap_io.read_streams(Path(path),mcap_io.schema_for(spec))
    stamps={r:np.asarray([t for t,_ in stream],np.int64) for r,stream in streams.items()}
    grid,selected=select_captures(stamps,spec)
    text=subprocess.check_output(['git','show','f8e8c1e:conversion/common/synchronization.py'],cwd=ROOT,text=True)
    old={'comparison_grid':grid}
    original='grid = np.arange(start, end + 1, PERIOD_NS, dtype=np.int64)'
    assert original in text
    exec(compile(text.replace(original,'grid = comparison_grid'),'baseline_solver_on_new_grid','exec'),old)
    _,reference=old['select_captures'](stamps,spec)
    result=dict(frames=len(grid),grid='new all-stream intersection; identical targets for both selectors',
                changed_source_indices={r:int(np.sum(selected[r]!=reference[r])) for r in spec.roles},
                changed_cameras={c:int(np.sum(selected[c]!=reference[c])) for c in spec.cameras})
    Path(destination).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':main()
