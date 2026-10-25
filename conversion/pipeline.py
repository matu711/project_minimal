"""Shared MCAP/raw -> synchronized absolute actions -> verified LeRobot pipeline."""
import importlib
import json
from pathlib import Path
from time import perf_counter
import uuid
from conversion.common.episode import Timings
from conversion.common.image_utils import resize_size
from conversion.common.fps import resolve_fps, consistent_fps, recorded_metadata
from conversion.common.timing_diagnostics import record_timing
from conversion.common import mcap_io, raw_reader, validation, lerobot_writer


def robot_spec(robot):
    if robot not in ('single', 'bimanual'):
        raise ValueError(f'Unknown robot: {robot}')
    return importlib.import_module(f'conversion.{robot}.adapter').SPEC


def is_raw(path):
    if not path.is_dir():
        return False
    return ((path/'dataset_meta.json').exists() or (path/'data.npz').exists()
            or (path/'frames.csv').exists() or any(p.is_dir() and
                ((p/'data.npz').exists() or (p/'frames.csv').exists()) for p in path.iterdir()))


def convert(robot, inputs, output, repo_id, task=None, schema_path=None, validate_only=False, single_options=None, resize=None, timing_log=None):
    resize = resize_size(resize)
    if robot == "single":
        from conversion.single.compatibility import convert as convert_single
        options = list(single_options or [])
        if resize is not None:
            options += ['--resize', *map(str, resize)]
        if timing_log is not None:
            options += ['--timing-log', str(timing_log)]
        return convert_single(inputs, output, repo_id, task, schema_path, validate_only, options)
    if single_options:
        raise ValueError("Legacy single options are not supported for bimanual conversion")
    started = perf_counter()
    spec, timings = robot_spec(robot), Timings()
    inputs, output = [Path(p).resolve() for p in inputs], Path(output).resolve()
    for source in inputs:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f'Output must be separate from input: {source}')
    if output.exists():
        raise FileExistsError(f'Output exists; choose a new dataset name: {output}')
    flags = [is_raw(path) for path in inputs]
    if any(flags) and not all(flags):
        raise ValueError('Do not mix raw bundles and MCAP-only inputs')
    raw = all(flags)
    if raw and schema_path is not None:
        raise ValueError('--mcap-schema applies only to MCAP-only inputs')
    timing = None
    if raw:
        sources = raw_reader.raw_episodes(inputs, spec)
        fps = consistent_fps(resolve_fps(dataset, [meta], str(p)) for p, dataset, meta in sources)
        timing = record_timing([p for p, _, _ in sources], fps, output, timing_log, robot)
        episodes = (raw_reader.read_episode(p, dataset, meta, spec, task, timings) for p, dataset, meta in sources)
    else:
        schema = mcap_io.schema_for(spec, schema_path)
        sources = mcap_io.source_files(inputs)
        fps = consistent_fps(resolve_fps(dataset, [meta], str(p)) for p in sources for dataset, meta in [recorded_metadata(p)])
        episodes = (mcap_io.read_episode(p, spec, schema, task, timings) for p in sources)
    if validate_only:
        summaries = []
        from PIL import Image
        import io
        for episode in episodes:
            validation.check_episode(episode, spec)
            for values in episode.images.values():
                for value in values:
                    with Image.open(io.BytesIO(value) if isinstance(value, bytes) else value) as image:
                        image.load()
            summaries.append(dict(source=episode.source, frames=len(episode.states)))
        return dict(status='input_validated', episodes=summaries, timings_s=timings.values)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f'.{output.name}.partial-{uuid.uuid4().hex[:8]}')
    try:
        summaries = lerobot_writer.write(staging, repo_id, spec, episodes, timings, resize)
        with timings.measure('verification'):
            report = validation.verify(staging, spec, summaries)
        report.update(status='verified', pipeline='recorded_raw_or_nearest_mcap_v2', robot=robot, fps=fps,
                      action_semantics='absolute joint rad and gripper m', sources=summaries,
                      timings_s=timings.values, total_s=perf_counter()-started)
        (staging/'provenance').mkdir(exist_ok=True)
        if timing is not None:
            (staging/'provenance/raw_timing.json').write_text(json.dumps(timing, indent=2)+'\n')
        (staging/'provenance/conversion.json').write_text(json.dumps(report, indent=2)+'\n')
        if output.exists():
            raise FileExistsError(f'Output appeared during conversion: {output}')
        staging.rename(output)
        return report
    except Exception as error:
        error.add_note(f'No completed dataset published. Partial output, if created: {staging}')
        raise
