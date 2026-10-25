#!/usr/bin/env python3
"""Historical ec58a7a restoration comparison, before explicit task simplification.

The old main() exercises retired --input-task/--task-replace options. For the
current contract run verify_conversion_alignment.py; compare() remains shared.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def module(name,path,replace=None):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    if replace:
        text=path.read_text().replace(*replace)
        exec(compile(text,str(path),'exec'),result.__dict__)
    else:
        spec.loader.exec_module(result)
    return result


def old_main(old,args):
    previous=sys.argv
    try:
        sys.argv=['reference',*map(str,args)]
        old.main()
    finally:
        sys.argv=previous


def compare(before,after):
    import pyarrow.parquet as pq
    info_a=json.loads((before/'meta/info.json').read_text())
    info_b=json.loads((after/'meta/info.json').read_text())
    assert info_a==info_b,'LeRobot info differs'
    for name in ['episodes.jsonl','tasks.jsonl','episodes_stats.jsonl']:
        assert (before/'meta'/name).read_bytes()==(after/'meta'/name).read_bytes(),name
    images=[]
    for path in sorted((before/'data').rglob('*.parquet')):
        a=pq.read_table(path).to_pydict()
        b=pq.read_table(after/path.relative_to(before)).to_pydict()
        assert a.keys()==b.keys()
        for key in a:
            if isinstance(a[key][0],str):assert a[key]==b[key]
            else:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    for video in sorted((before/'videos').rglob('*.mp4')):
        other=after/video.relative_to(before)
        if video.read_bytes()==other.read_bytes():
            images.append(dict(video=str(video.relative_to(before)),encoded_file_exact=True))
            continue
        def decode(path):
            return subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(path),
                '-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
        a,b=decode(video),decode(other)
        try:
            while True:
                x=a.stdout.read(1024*1024);y=b.stdout.read(1024*1024)
                assert x==y,f'Decoded video pixels differ: {video}'
                if not x:break
            assert a.wait()==b.wait()==0
        finally:
            a.stdout.close();b.stdout.close()
            if a.poll() is None:a.kill();a.wait()
            if b.poll() is None:b.kill();b.wait()
        images.append(dict(video=str(video.relative_to(before)),encoded_file_exact=False,all_decoded_pixels_exact=True))
    return dict(frames=info_a['total_frames'],episodes=info_a['total_episodes'],
                all_parquet_fields_exact=True,metadata_exact=True,videos=images)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    old=module('single_reference_converter',args.reference/'conversion/single/convert.py')
    old_mcap=module('single_reference_mcap',args.reference/'conversion/mcap_io.py')
    old_extract=module('single_reference_extract',args.reference/'conversion/single/extract.py',
                       ('from conversion.mcap_io import','from single_reference_mcap import'))
    from conversion.pipeline import convert
    from tests.conversion_fixtures import record,change_npz
    source=args.source.resolve()
    raw=root/'raw_input';raw.mkdir()
    (raw/'dataset_meta.json').write_bytes((source/'dataset_meta.json').read_bytes())
    (raw/'20260825_224317').symlink_to(source/'20260825_224317',target_is_directory=True)
    results=[]
    def run(name,inputs,options=None,task=None):
        before,after=root/('reference_'+name),root/('restored_'+name)
        print('BEGIN',name,flush=True);start=time.perf_counter()
        command=[]
        for p in inputs:command+=['--input',str(p)]
        old_main(old,[*command,'--output',str(before),'--repo-id','local/compat',*(options or [])])
        old_seconds=time.perf_counter()-start;start=time.perf_counter()
        result=convert('single',inputs,after,'local/compat',task=task,single_options=options)
        check=compare(before,after)
        check.update(case=name,reference_s=old_seconds,restored_s=time.perf_counter()-start,
                     timing_log=result['raw_timing_log'])
        results.append(check)
        (root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
        print('PASS',name,check['frames'],flush=True)
    run('raw_450',[raw])
    # Reference MCAP extraction is the original code and original decoder.
    mcap=source/'20260825_224317/20260825_224317.mcap'
    extracted=old_extract.extract([mcap],root/'reference_mcap_raw','pick up the white object')
    before=root/'reference_mcap_503';after=root/'restored_mcap_503'
    print('BEGIN mcap_503',flush=True)
    old_main(old,['--input',str(extracted),'--output',str(before),'--repo-id','local/compat'])
    result=convert('single',[mcap],after,'local/compat',task='pick up the white object')
    check=compare(before,after);check.update(case='mcap_503',timing_log=result['raw_timing_log']);results.append(check)
    (root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PASS mcap_503',flush=True)
    inputs=[]
    for label in ['ee_a','ee_b']:
        path=record(root/label,'single')
        poses=np.zeros((4,8),np.float32);poses[:,:3]=np.arange(4)[:,None]*np.array([.01,.02,.03]);poses[:,6]=[1,-1,1,1];poses[:,7]=[.01,.02,.03,.04]
        change_npz(path,observation_ee=poses)
        meta=json.loads((path.parent/'dataset_meta.json').read_text())
        meta.update(format='piper_raw_fixed_rate_synced_v6_ee_pose',ee_observation_dim=8,
                    observation_ee_layout=['x','y','z','qx','qy','qz','qw','gripper'])
        (path.parent/'dataset_meta.json').write_text(json.dumps(meta))
        if label=='ee_a':
            ep=json.loads((path/'meta.json').read_text());ep['successful']=False;(path/'meta.json').write_text(json.dumps(ep))
        inputs.append(path.parent)
    options=['--include-failed','--input-task','Pick A','--input-task','Pick C','--task-replace','A=B','--task-replace','C=D','--resize','32','24']
    run('ee_absolute',inputs,options)
    run('ee_delta',inputs,options+['--action-mode','ee_delta_axis_angle'])
    # Acceptance comparison for every existing episode; no LeRobot outputs.
    old_main(old,['--input',str(source),'--output',str(root/'unused_reference'),'--validate-only'])
    validated=convert('single',[source],root/'unused_restored','local/compat',validate_only=True)
    (root/'acceptance.json').write_text(json.dumps(validated,indent=2)+'\n')
    print('ALL PASS',len(results),'comparisons;',len(validated['episodes']),'real raw episodes accepted',flush=True)


if __name__=='__main__':main()
