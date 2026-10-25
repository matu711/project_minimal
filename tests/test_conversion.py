"""Shared pipeline correctness, strict failures, atomic publication and reader compatibility."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from conversion.pipeline import convert, robot_spec
from conversion.common import mcap_io, raw_reader, validation
from conversion.common.episode import Timings
from conversion.common.synchronization import nearest_indices, select_captures
from tests.conversion_fixtures import record, change_npz


def streams(robot):
    spec = robot_spec(robot)
    origin = 1_792_033_200_000_000_000
    result = {}
    for i, role in enumerate(spec.roles):
        result[role] = [(origin+j*50_000_000, np.arange(7,dtype=np.float32)+i*10+j/100) for j in range(4)]
    for i, camera in enumerate(spec.cameras):
        payload = io.BytesIO()
        Image.new('RGB',(64,48) if robot == 'single' else (64+16*i,48+16*i),(20+i*70,30,60)).save(payload,format='JPEG')
        result[camera] = [(origin+j*50_000_000,payload.getvalue()) for j in range(4)]
    return result


@pytest.mark.parametrize('robot',['single','bimanual'])
@pytest.mark.parametrize('kind',['mcap','raw'])
def test_full_conversion_and_real_reader(tmp_path, monkeypatch, robot, kind):
    import pyarrow.parquet as pq
    spec = robot_spec(robot)
    if kind == 'mcap':
        source = tmp_path/'record.mcap'
        source.write_bytes(b'mocked decoder; real writer')
        data = streams(robot)
        monkeypatch.setattr(mcap_io,'read_streams',lambda *_:data)
    else:
        source = tmp_path/'raw'
        episode = record(source,robot)
        with np.load(episode/'data.npz') as d:
            expected = d['observation_state'].copy()
        if robot == 'bimanual':
            expected = expected[:,[*range(7,14),*range(7)]]
        # A raw conversion must never open MCAP or recover rows.
        monkeypatch.setattr(mcap_io,'read_streams',lambda *_:pytest.fail('Unexpected raw MCAP recovery'))
    output = tmp_path/'out'
    report = convert(robot,[source],output,'local/test',task='Task.' if kind == 'mcap' else None)
    assert report['frames'] == 4 and report['episodes'] == 1
    table = pq.read_table(next((output/'data').rglob('*.parquet'))).to_pydict()
    np.testing.assert_allclose(table['timestamp'],np.arange(4)/20)
    if kind == 'raw':
        np.testing.assert_array_equal(table['observation.state'],expected)
        assert report['sources'][0]['task'] == '  Stack cubes.\nKeep order.  '
    else:
        parts = [np.stack([v for _,v in data[r]]) for r in spec.roles if r.endswith('state')]
        np.testing.assert_array_equal(table['observation.state'],np.concatenate(parts,axis=1))
    quality = [key for key in table if key.startswith('quality.')]
    assert quality == (['quality.frame_valid'] if robot == 'bimanual' else [])
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    loaded = LeRobotDataset(repo_id='local/test',root=output,video_backend='pyav',
                            delta_timestamps={'action':[i/20 for i in range(3)]})
    for index in [0,2,3]:
        item = loaded[index]
        assert item['action'].shape == (3,spec.dim)
        for c in spec.cameras.values():
            image = item[f'observation.images.{c}']
            assert image.shape[0] == 3 and np.isfinite(image.numpy()).all()
    assert loaded[3]['action_is_pad'][1:].all()
    assert json.loads((output/'provenance/conversion.json').read_text())['status'] == 'verified'
    with pytest.raises(FileExistsError):
        convert(robot,[source],output,'local/test')


@pytest.mark.parametrize('robot',['bimanual'])
def test_raw_gap_missing_image_and_bad_vectors(tmp_path,robot):
    path = record(tmp_path/'raw',robot)
    spec = robot_spec(robot)
    change_npz(path,timestamp=np.array([0,.05,.2,.25]))
    result = raw_reader.read_episode(path,{},json.loads((path/'meta.json').read_text()),spec,None,Timings())
    assert len(result.states) == 4
    change_npz(path,timestamp=np.arange(4)/20,observation_state=np.full((4,spec.dim),np.nan,np.float32))
    with pytest.raises(ValueError,match='finite'):
        convert(robot,[path.parent],tmp_path/'bad','local/bad')
    assert not (tmp_path/'bad').exists()
    change_npz(path,observation_state=np.zeros((4,spec.dim),np.float32))
    next((path/'front').glob('*.png')).unlink()
    with pytest.raises(ValueError,match='missing or external image'):
        convert(robot,[path.parent],tmp_path/'missing','local/missing')


def test_nearest_tie_and_boundaries():
    np.testing.assert_array_equal(nearest_indices([0,100],np.array([50]),50,'camera'),[0])
    nearest_indices([0],np.array([100_000_000]),100_000_000,'camera')
    with pytest.raises(ValueError,match='100ms'):
        nearest_indices([0],np.array([100_000_001]),100_000_000,'camera')
    with pytest.raises(ValueError,match='30ms'):
        nearest_indices([0],np.array([30_000_001]),30_000_000,'robot')


@pytest.mark.parametrize('robot',['single','bimanual'])
def test_missing_robot_camera_and_group_span(robot):
    spec = robot_spec(robot)
    values = streams(robot)
    stamps = {r:np.array([t for t,_ in v]) for r,v in values.items()}
    origin = stamps[spec.roles[0]][0]
    # All individual captures are within 30ms, but their span is 46ms.
    for i,r in enumerate(spec.roles):
        stamps[r] = np.array([origin-23_000_000 if i%2 else origin+23_000_000,
                             origin+150_000_000+(23_000_000 if i%2 else -23_000_000)])
    with pytest.raises(ValueError):
        select_captures(stamps,spec)
    with pytest.raises(ValueError,match='missing or unordered'):
        nearest_indices([],np.array([origin]),30_000_000,'robot')


def test_grid_endpoints_preserve_robot_specific_range():
    counts = []
    for robot in ['single','bimanual']:
        spec = robot_spec(robot)
        stamps = {r:np.array([t for t,_ in v]) for r,v in streams(robot).items()}
        for c in spec.cameras:
            stamps[c] = stamps[c][1:]
        grid,_ = select_captures(stamps,spec)
        counts.append(len(grid))
    assert counts == [3,3]


@pytest.mark.parametrize('robot',['single','bimanual'])
def test_failure_does_not_publish_output(tmp_path,monkeypatch,robot):
    source = tmp_path/'raw'
    record(source,robot)
    def fail(*_):
        raise ValueError('forced output verification failure')
    if robot == 'single':
        from conversion.single import verify as single_verifier
        monkeypatch.setattr(single_verifier,'verify',fail)
    else:
        monkeypatch.setattr(validation,'verify',fail)
    with pytest.raises(ValueError,match='forced output'):
        convert(robot,[source],tmp_path/'out','local/test')
    assert not (tmp_path/'out').exists()
    assert len(list(tmp_path.glob('.out.partial-*'))) == 1


@pytest.mark.parametrize('problem',['type','names','dimension','finite','missing','image'])
def test_ros_schema_and_payload_validation(tmp_path,monkeypatch,problem):
    from mcap import reader
    spec = robot_spec('single')
    schema = mcap_io.schema_for(spec)
    stamp = SimpleNamespace(secs=1,nsecs=0)
    payload = SimpleNamespace(header=SimpleNamespace(stamp=stamp),name=schema['joint_names'],position=np.zeros(7))
    roles = {**schema['robot_topics'],**schema['camera_topics']}
    messages=[]
    for role,topic in roles.items():
        robot = role in spec.roles
        value = payload if robot else SimpleNamespace(header=payload.header,data=b'not-empty')
        typename = 'sensor_msgs/JointState' if robot else 'sensor_msgs/CompressedImage'
        if role == 'state':
            if problem == 'missing':continue
            if problem == 'type':typename='sensor_msgs/Image'
            if problem == 'names':value=SimpleNamespace(**{**vars(payload),'name':['wrong']*7})
            if problem == 'dimension':value=SimpleNamespace(**{**vars(payload),'position':[1,2]})
            if problem == 'finite':value=SimpleNamespace(**{**vars(payload),'position':[np.nan]*7})
        if not robot and problem == 'image':value=SimpleNamespace(header=payload.header,data=b'')
        messages.append((SimpleNamespace(name=typename),SimpleNamespace(topic=topic),None,value))
    monkeypatch.setattr(reader,'make_reader',lambda *a,**k:SimpleNamespace(iter_decoded_messages=lambda **_:iter(messages)))
    source = tmp_path/'test.mcap';source.touch()
    with pytest.raises(ValueError):mcap_io.read_streams(source,schema)


@pytest.mark.parametrize('robot',['bimanual'])
@pytest.mark.parametrize('problem',['robot_capture','camera_capture','shape','corrupt'])
def test_sync_and_image_failures_never_publish(tmp_path,robot,problem):
    path=record(tmp_path/'raw',robot)
    spec=robot_spec(robot)
    if problem in ('robot_capture','camera_capture'):
        with np.load(path/'data.npz') as data:
            key=(spec.roles[0] if problem=='robot_capture' else 'front')+'_ros_timestamp'
            stamps=data[key].copy()
        stamps += .031 if problem=='robot_capture' else .151
        change_npz(path,**{key:stamps})
    else:
        image=sorted((path/'front').glob('*.png'))[1]
        if problem=='corrupt':image.write_bytes(b'broken image')
        else:Image.new('RGB',(100,60)).save(image)
    if problem in ('robot_capture','camera_capture'):
        assert convert(robot,[path.parent],tmp_path/'out','local/bad')['frames'] == 4
    else:
        with pytest.raises(ValueError):
            convert(robot,[path.parent],tmp_path/'out','local/bad')
        assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('configured',[20,10])
def test_single_legacy_uses_dataset_fps_not_episode_measurement(tmp_path,configured):
    path=record(tmp_path/'raw','single')
    meta=json.loads((path/'meta.json').read_text())
    meta.update(fps=19.7,configured_fps=configured)
    (path/'meta.json').write_text(json.dumps(meta))
    result=convert('single',[path.parent],tmp_path/'out','local/test',validate_only=True)
    assert result['status']=='input_validated'
