"""Legacy single-arm contract: irregular clocks, EE/resize/task/config and selection."""
import json
from pathlib import Path
import numpy as np
import pytest
from conversion.pipeline import convert
from conversion.single import convert as legacy
from tests.conversion_fixtures import record, change_npz


def metadata(path,**updates):
    value=json.loads(path.read_text());value.update(updates);path.write_text(json.dumps(value))


def test_irregular_raw_keeps_all_samples_and_reports_real_clock(tmp_path):
    path=record(tmp_path/'raw','single')
    times=np.array([.01,.06,.21,.26])
    change_npz(path,timestamp=times,unix_timestamp=1000+times)
    # Legacy single ignores CSV and capture errors and keeps per-row images.
    (path/'frames.csv').write_text('not part of the legacy input contract\n')
    change_npz(path,state_ros_timestamp=1000+times+.2)
    with pytest.warns(RuntimeWarning,match='irregular intervals'):
        result=convert('single',[path.parent],tmp_path/'out','local/legacy')
    import pyarrow.parquet as pq
    table=pq.read_table(next((tmp_path/'out/data').rglob('*.parquet'))).to_pydict()
    np.testing.assert_allclose(table['timestamp'],np.arange(4)/20)
    with np.load(path/'data.npz') as raw:
        np.testing.assert_array_equal(table['observation.state'],raw['observation_state'])
        np.testing.assert_array_equal(table['action'],raw['action'])
    audit=json.loads((tmp_path/'out/provenance/raw_timing.json').read_text())
    clock=audit['episodes'][0]['clocks']['timestamp']
    np.testing.assert_array_equal(clock['timestamps_s'],times)
    assert clock['irregular_intervals'][0]['row_from']==1
    assert clock['irregular_intervals'][0]['actual_delta_s']==pytest.approx(.15)
    assert 'NOT measured' in audit['output_clock']
    assert Path(result['raw_timing_log']).is_file()


def test_ee_delta_resize_and_explicit_task_roundtrip(tmp_path):
    path=record(tmp_path/'raw','single')
    poses=np.zeros((4,8),np.float32)
    poses[:,:3]=np.arange(4)[:,None]*np.array([.01,.02,.03])
    poses[:,6]=[1,-1,1,1]
    poses[:,7]=[.01,.02,.03,.04]
    change_npz(path,observation_ee=poses)
    metadata(path.parent/'dataset_meta.json',format='piper_raw_fixed_rate_synced_v6_ee_pose',
             ee_observation_dim=8,observation_ee_layout=['x','y','z','qx','qy','qz','qw','gripper'])
    result=convert('single',[path.parent],tmp_path/'out','local/ee',
                   task='Pick B', single_options=['--action-mode','ee_delta_axis_angle','--resize','32','24'])
    assert result['frames']==3
    import pyarrow.parquet as pq
    table=pq.read_table(next((tmp_path/'out/data').rglob('*.parquet'))).to_pydict()
    _,expected,_=legacy.make_ee_delta_actions(poses)
    np.testing.assert_array_equal(table['action'],expected)
    assert table['observation.ee_pose']
    info=json.loads((tmp_path/'out/meta/info.json').read_text())
    assert info['features']['observation.images.front']['shape']==[24,32,3]
    assert json.loads((tmp_path/'out/meta/tasks.jsonl').read_text())['task']=='Pick B'


@pytest.mark.parametrize('format',['piper_raw_v1','piper_raw_fixed_rate_synced_v5','piper_raw_fixed_rate_synced_v6_ee_pose'])
def test_supported_formats_missing_clocks_and_single_frame_are_accepted(tmp_path,format):
    path=record(tmp_path/'raw','single',count=1)
    metadata(path.parent/'dataset_meta.json',format=format)
    with np.load(path/'data.npz') as d:
        np.savez(path/'no-clock.npz',observation_state=d['observation_state'],action=d['action'])
    (path/'no-clock.npz').replace(path/'data.npz')
    with pytest.warns(RuntimeWarning,match='No usable measured frame clock'):
        report=convert('single',[path.parent],tmp_path/'out','local/test',validate_only=True)
    assert report['episodes'][0]['frames']==1


def test_task_precedence_and_fail_selection(tmp_path):
    path=record(tmp_path/'first','single')
    metadata(path/'meta.json',task='',tsku='Ignored legacy alias',successful=False)
    with pytest.raises(RuntimeError,match='No episodes selected'):
        convert('single',[path.parent],tmp_path/'skipped','local/test',validate_only=True)
    result=convert('single',[path.parent],tmp_path/'included','local/test',task=' Explicit ',
                   single_options=['--include-failed'])
    assert result['sources'][0]['task']==' Explicit '
    for option in ['--input-task','--task-replace']:
        with pytest.raises(SystemExit):
            legacy.parse_args(['--input',str(path.parent),option,'obsolete'])


def test_config_path_repo_and_task_inference(tmp_path):
    config=tmp_path/'config.yaml'
    config.write_text(f'data_collection:\n  save_dir: {tmp_path}/data\n  dataset_repo_id: local/from_config\n  task: Config fallback\n')
    output,repo=legacy.infer_output_root(explicit_output=None,explicit_repo_id=None,config_path=config,dataset_meta={})
    assert output==tmp_path/'data/local/from_config' and repo=='local/from_config'
    args=legacy.parse_args(['--input',str(tmp_path),'--config',str(config),'--include-failed','--overwrite'])
    assert args.overwrite and args.include_failed


def test_shared_cli_forwards_single_options_and_rejects_them_for_dual(tmp_path,monkeypatch):
    from conversion import cli
    seen={}
    def run(*args,**kwargs):
        seen.update(kwargs)
        return {'status':'ok'}
    monkeypatch.setattr(cli,'convert',run)
    assert cli.main(['--robot','single','--dataset','fixture','--resize','32','24','--action-mode','ee_delta_axis_angle'])==0
    assert seen['single_options']==['--action-mode','ee_delta_axis_angle']
    assert seen['resize']==[32,24]
    assert cli.main(['--robot','bimanual','--dataset','fixture','--resize','32','24'])==0


def test_single_mcap_schema_retains_legacy_list_requirement(tmp_path):
    from conversion.single.mcap_io import schema_for
    value=json.loads((Path(__file__).resolve().parents[1]/'configs/single/conversion.json').read_text())
    value['joint_names']='ABCDEFG'
    path=tmp_path/'schema.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='seven unique'):
        schema_for('single',path)


def test_explicit_timing_log_is_used_without_overwriting(tmp_path):
    path=record(tmp_path/'raw','single')
    log=tmp_path/'requested-clock.json'
    result=convert('single',[path.parent],tmp_path/'out','local/test',validate_only=True,
                   single_options=['--timing-log',str(log)])
    assert result['raw_timing_log']==str(log)
    assert log.is_file()
    assert not (tmp_path/'out').exists()
    original=log.read_bytes()
    with pytest.raises(FileExistsError):
        convert('single',[path.parent],tmp_path/'other','local/test',validate_only=True,
                single_options=['--timing-log',str(log)])
    assert log.read_bytes()==original
