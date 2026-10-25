#!/usr/bin/env python3
"""Package complete OpenPI checkpoints and transfer explicitly selected archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def local_path(value):
    path = Path(value).expanduser()
    container_root = Path('/workspace/project_minimal')
    if path.is_absolute() and path.is_relative_to(container_root):
        # Metadata is written inside Docker; the same tools can run on the host
        # to use its SSH configuration without mounting credentials into Docker.
        path = ROOT / path.relative_to(container_root)
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'File must be inside this project: {path}')
    return path


def select(args):
    if Path(args.experiment).name != args.experiment or args.experiment in ('.', '..'):
        raise ValueError('experiment must be a single directory name')
    records = list((ROOT / 'logs').glob(f'*/{args.experiment}/metadata.json'))
    if args.config_name:
        records = [p for p in records if p.parent.parent.name == args.config_name]
    if len(records) != 1:
        raise ValueError(f'Expected one experiment metadata.json, found {len(records)}. Use --config-name to disambiguate.')
    record = records[0]
    metadata = json.loads(record.read_text())
    directory = local_path(metadata['checkpoint_dir'])
    # Async Orbax writes temporary names; only numeric, finalized steps are eligible.
    steps = sorted((p for p in directory.iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name))
    steps = [p for p in steps if (p / 'params').is_dir() and (p / '_CHECKPOINT_METADATA').is_file()]
    if args.step is not None:
        steps = [p for p in steps if int(p.name) == args.step]
    if not steps:
        raise FileNotFoundError('No finalized Orbax checkpoint with params found')
    checkpoint = steps[-1]
    # Preserve optimizer and train_state too, so resume remains possible.
    files = [(p, f'checkpoint/{checkpoint.name}/{p.relative_to(checkpoint)}') for p in checkpoint.rglob('*') if p.is_file()]
    stats = local_path(metadata['norm_stats'])
    config = local_path(metadata['training_yaml'])
    files += [(record, 'metadata.json'), (config, 'training.yaml'), (stats, 'norm_stats.json')]
    for filename in ('tuning_norm_config.json', 'bimanual_norm_config.json'):
        if (stats.parent / filename).is_file():
            files.append((stats.parent / filename, filename))
    for source, _ in files:
        local_path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Expected regular file: {source}')
    archive = ROOT / 'exports' / f"{metadata['config']}-{args.experiment}-step{checkpoint.name}.tar.gz"
    return metadata, files, archive


def package(args):
    metadata, files, archive = select(args)
    print(json.dumps(dict(archive=str(archive), method=metadata['method'], base_params=metadata['base_params'],
                         files=[dict(source=str(p), archive=n, bytes=p.stat().st_size) for p,n in files]), indent=2))
    if args.dry_run:
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = archive.with_name(archive.name + '.sha256')
    if archive.exists() or checksum_path.exists():
        raise FileExistsError(f'Export already exists: {archive}')
    manifest = {name: sha256(path) for path,name in files}
    import io
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    with archive.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz', dereference=True) as tar:
            for source, name in files:
                tar.add(source, arcname=name, recursive=False)
            info = tarfile.TarInfo('MANIFEST.sha256.json')
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    # Verify what was actually archived, not just the original files.
    with tarfile.open(archive, 'r:gz') as tar:
        for name, expected in manifest.items():
            stream = tar.extractfile(name)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != expected:
                raise ValueError(f'Archive integrity check failed: {name}')
    checksum_path.write_text(f'{sha256(archive)}  {archive.name}\n')
    print(f'Verified: {archive}\nChecksum: {checksum_path}')
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['package', 'transfer'])
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--config-name')
    parser.add_argument('--step', type=int)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--destination')
    parser.add_argument('--transport', choices=['scp', 'rsync'], default='scp')
    args = parser.parse_args(argv)
    if args.operation == 'package':
        package(args)
        return
    if not args.destination or args.destination.startswith('-') or ':' not in args.destination or any(c.isspace() for c in args.destination):
        parser.error('transfer requires --destination user@host:/path (without whitespace)')
    _, _, archive = select(args)
    checksum = archive.with_name(archive.name + '.sha256')
    if not args.dry_run:
        if not archive.is_file() or not checksum.is_file():
            raise FileNotFoundError('Run package_checkpoints.sh first')
        if sha256(archive) != checksum.read_text().split()[0]:
            raise ValueError('Export SHA256 mismatch; transfer aborted')
    command = [args.transport]
    if args.transport == 'rsync':
        command += ['-av', '--protect-args']
    command += ['--', str(archive), str(checksum), args.destination]
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
