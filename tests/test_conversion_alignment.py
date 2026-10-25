"""Specification alignment: preserve raw rows, independent MCAP nearest, explicit tasks/FPS."""
import json
import numpy as np
import pytest
from conversion.common.fps import resolve_fps
from conversion.common.task import resolve_task
from conversion.common.image_utils import load_rgb
from conversion.common.synchronization import select_captures
from conversion.common import raw_reader
from conversion.common.episode import Timings
from conversion.pipeline import convert, robot_spec
from tests.conversion_fixtures import record, change_npz


def test_fps_precedence_conflict_and_default():
    assert resolve_fps({'fps':20.9},[{'fps':17.8}])==20
    assert resolve_fps({},[{'fps':15},{'fps':15}])==15
    with pytest.raises(ValueError,match='conflicting'):
        resolve_fps({},[{'fps':15},{'fps':20}])
    with pytest.warns(RuntimeWarning,match='20 FPS'):
        assert resolve_fps({},[])==20
    for value in [0,-1,float('nan'),'bad']:
        with pytest.raises(ValueError):resolve_fps({'fps':value})


def test_simple_task_contract():
    assert resolve_task(' Override ',{'task':'Episode'},{'task':'Dataset'})==' Override '
    assert resolve_task(None,{'task':' Episode '},{'task':'Dataset'})==' Episode '
    assert resolve_task(None,{}, {'task':'Dataset'})=='Dataset'
    with pytest.raises(ValueError,match='missing recorded task'):
        resolve_task(None,{'tsku':'Legacy'}, {'config_task':'Old fallback'})
    with pytest.raises(ValueError,match='nonempty'):
        resolve_task('',{'task':'valid'}, {})


def test_dual_independent_nearest_does_not_search_alternative_groups():
    spec=robot_spec('bimanual')
    # At target 50ms nearest robot captures are 26 and 74ms (48ms span).
    # Alternative captures at 79ms could form a valid group; substitution is forbidden.
    stamps={r:np.array([0,26_000_000,79_000_000,100_000_000]) for r in spec.roles}
    stamps[spec.roles[1]]=np.array([0,74_000_000,100_000_000])
    stamps.update({c:np.array([0,50_000_000,100_000_000]) for c in spec.cameras})
    with pytest.raises(ValueError,match='capture span exceeds 30ms'):
        select_captures(stamps,spec)


def test_dual_irregular_raw_keeps_csv_images_and_all_rows(tmp_path):
    path=record(tmp_path/'raw')
    change_npz(path,timestamp=np.array([0,.05,.2,.25]),frame_index=np.array([0,1,4,5]))
    # Recorder CSV and NPZ retain matching, discontinuous IDs.
    import csv
    with (path/'frames.csv').open() as f:rows=list(csv.DictReader(f)); fields=list(rows[0])
    for row,index in zip(rows,[0,1,4,5]):row['frame_index']=str(index)
    with (path/'frames.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    episode=raw_reader.read_episode(path,{},json.loads((path/'meta.json').read_text()),robot_spec('bimanual'),None,Timings())
    assert len(episode.states)==4
    assert episode.images['front']==[(path/row['front_image']).resolve() for row in rows]
    with pytest.warns(RuntimeWarning,match='irregular intervals'):
        report=convert('bimanual',[path.parent],tmp_path/'out','local/test',resize=(32,24))
    assert report['frames']==4
    timing=json.loads((tmp_path/'out/provenance/raw_timing.json').read_text())
    assert timing['episodes'][0]['clocks']['timestamp']['irregular_intervals'][0]['actual_delta_s']==pytest.approx(.15)


@pytest.mark.parametrize('robot',['single','bimanual'])
def test_metadata_fps_and_shared_resize_match_legacy_pixels(tmp_path,robot):
    path=record(tmp_path/'raw',robot)
    meta=json.loads((path.parent/'dataset_meta.json').read_text());meta['fps']=10.9
    (path.parent/'dataset_meta.json').write_text(json.dumps(meta))
    with pytest.warns(RuntimeWarning,match='irregular intervals'):
        report=convert(robot,[path.parent],tmp_path/'out','local/test',resize=(32,24),task='Explicit')
    assert report['fps']==10
    import pyarrow.parquet as pq
    data=pq.read_table(next((tmp_path/'out/data').rglob('*.parquet'))).to_pydict()
    np.testing.assert_allclose(data['timestamp'],np.arange(4)/10)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    loaded=LeRobotDataset(repo_id='local/test',root=tmp_path/'out',video_backend='pyav')
    assert loaded.fps==10 and len(loaded)==4
    assert loaded[3]['observation.state'].shape==(7 if robot=='single' else 14,)
    info=json.loads((tmp_path/'out/meta/info.json').read_text())
    for camera in robot_spec(robot).cameras.values():
        assert info['features'][f'observation.images.{camera}']['shape']==[24,32,3]
    from PIL import Image
    image=next((path/'front').iterdir())
    expected=np.asarray(Image.open(image).convert('RGB').resize((32,24),Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(load_rgb(image,(32,24)),expected)


@pytest.mark.parametrize('robot',['single','bimanual'])
def test_mcap_grid_uses_recorded_fps_and_task(tmp_path,monkeypatch,robot):
    from tests.test_conversion import streams
    from conversion.common import mcap_io
    source=tmp_path/'record.mcap';source.write_bytes(b'mocked decoder')
    (tmp_path/'meta.json').write_text(json.dumps({'fps':10,'task':'Recorded MCAP task'}))
    monkeypatch.setattr(mcap_io,'read_streams',lambda *_:streams(robot))
    report=convert(robot,[source],tmp_path/'out','local/test')
    assert report['fps']==10 and report['frames']==2
    assert report['sources'][0]['task']=='Recorded MCAP task'


def test_dual_raw_without_measured_clocks_is_warned_not_rejected(tmp_path):
    path=record(tmp_path/'raw')
    with np.load(path/'data.npz') as data:
        arrays={key:data[key] for key in data.files if 'timestamp' not in key}
    np.savez(path/'data.npz',**arrays)
    with pytest.warns(RuntimeWarning,match='No usable measured frame clock'):
        result=convert('bimanual',[path.parent],tmp_path/'out','local/test',validate_only=True)
    assert result['episodes'][0]['frames']==4
