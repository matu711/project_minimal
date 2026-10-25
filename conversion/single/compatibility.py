"""Preserve the original single-arm converter behind the shared public pipeline."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
import uuid


def convert(inputs, output, repo_id, task=None, schema_path=None, validate_only=False, options=None):
    from conversion.single import convert as legacy, extract, verify
    options = list(options or [])
    if '--overwrite' in options:
        raise ValueError('Shared CLI disables --overwrite; use a new output (legacy direct CLI retains explicit overwrite)')
    inputs, output = [Path(p).resolve() for p in inputs], Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists; choose a new dataset name: {output}')
    for source in inputs:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f'Output must be separate from input: {source}')
    started = perf_counter()
    token = uuid.uuid4().hex[:8]
    staging = output.with_name(f'.{output.name}.partial-{token}')
    timing_log = output.with_name(f'{output.name}.raw_timing-{token}.json')
    flags = [(p/'dataset_meta.json').is_file() or (p/'data.npz').is_file() for p in inputs]
    if any(flags) and not all(flags):
        raise ValueError('Do not mix raw datasets and MCAP-only inputs')
    if all(flags) and schema_path:
        raise ValueError('--mcap-schema is only used for MCAP-only input')
    with ExitStack() as stack:
        raw_roots = []
        if not all(flags):
            raw_roots = [extract.extract(inputs, output.with_name(output.name+'.raw'), task, schema_path)]
        else:
            for source in inputs:
                if (source/'dataset_meta.json').is_file():
                    raw_roots.append(source)
                elif (source.parent/'dataset_meta.json').is_file():
                    # Convenience for one episode; metadata and all samples remain unchanged.
                    root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='single-raw-input-')))
                    shutil.copyfile(source.parent/'dataset_meta.json', root/'dataset_meta.json')
                    (root/source.name).symlink_to(source, target_is_directory=True)
                    raw_roots.append(root)
                else:
                    raise ValueError(f'{source}: legacy single raw requires dataset_meta.json in input or parent')
        command = []
        for path in raw_roots:
            command += ['--input',str(path)]
        command += ['--output',str(staging), '--repo-id',repo_id,'--timing-log',str(timing_log)]
        if task is not None:
            command += ['--task',task]
        if validate_only:
            command += ['--validate-only']
        command += options
        parsed = legacy.parse_args(command)
        timing_log = parsed.timing_log.resolve()
        if parsed.output.resolve() != staging or parsed.overwrite:
            raise ValueError('Single options must not override the shared output or overwrite protection')
        try:
            legacy.main(command)
            timing = json.loads(timing_log.read_text())
            if validate_only:
                return dict(status='input_validated',episodes=[dict(source=e['source'],frames=e['raw_frames']) for e in timing['episodes']],
                            raw_timing_log=str(timing_log),timings_s={'single_legacy_total':perf_counter()-started})
            report = verify.verify(staging)
            info = json.loads((staging/'meta/info.json').read_text())
            metadata = [json.loads(s) for s in (staging/'meta/episodes.jsonl').read_text().splitlines()]
            report.update(status='verified',pipeline='single_legacy_compatible_v1',robot='single',fps=info['fps'],
                          timestamp_policy='nominal frame_index/fps; raw clocks are separately logged',
                          action_mode=parsed.action_mode,raw_timing_log=str(timing_log),
                          sources=[dict(task=e['tasks'][0],frames=e['length']) for e in metadata],
                          timings_s={'single_legacy_total':perf_counter()-started})
            (staging/'provenance/conversion.json').write_text(json.dumps(report,indent=2)+'\n')
            if output.exists():
                raise FileExistsError(f'Output appeared during conversion: {output}')
            staging.rename(output)
            return report
        except Exception as error:
            error.add_note(f'No completed dataset published. Partial output, if created: {staging}')
            raise
