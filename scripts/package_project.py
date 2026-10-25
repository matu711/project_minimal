#!/usr/bin/env python3
"""Create a code-only, independently buildable project_minimal archive."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ('build.sh', 'run.sh', 'Dockerfile', 'compose.yaml', '.dockerignore', '.gitignore',
             'requirements.lock', 'README.md', 'VALIDATION.md', 'SOURCE_PROVENANCE.md', 'CONVERSION_SIMPLIFICATION.md', 'SINGLE_COMPATIBILITY_RESTORE.md', 'CONVERSION_ALIGNMENT.md')
FOLDERS = ('conversion', 'training', 'configs', 'scripts', 'tests',
           'openpi/src', 'openpi/packages/openpi-client')
OPENPI_FILES = ('openpi/pyproject.toml', 'openpi/README.md', 'openpi/scripts/__init__.py',
                'openpi/scripts/compute_norm_stats.py', 'openpi/custom_scripts/train.py')


def files_to_share():
    candidates = [ROOT / name for name in (*TOP_LEVEL, *OPENPI_FILES)]
    candidates += list((ROOT / 'openpi').glob('LICENSE*'))
    for folder in FOLDERS:
        candidates.extend((ROOT / folder).rglob('*'))
    result = []
    for path in sorted(set(candidates)):
        relative = path.relative_to(ROOT)
        if any(part.startswith('.') for part in relative.parts) and str(relative) not in ('.gitignore', '.dockerignore'):
            continue
        if '__pycache__' in relative.parts or path.is_symlink() or not path.is_file():
            continue
        if relative.parts[0] == 'openpi' and (path.name.endswith('_test.py') or path.name == 'conftest.py'):
            continue
        if (str(relative) in TOP_LEVEL or str(relative) in OPENPI_FILES or path.name.startswith('LICENSE')
                or path.suffix in ('.py', '.pyi', '.md', '.sh', '.toml', '.json', '.yaml', '.yml', '.typed')):
            result.append(path)
    for filename in (*TOP_LEVEL, *OPENPI_FILES):
        if ROOT / filename not in result:
            raise FileNotFoundError(f'Required share source missing: {filename}')
    return result


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'exports/project_minimal-source.tar.gz')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    files = files_to_share()
    output = args.output.expanduser().resolve()
    print(json.dumps(dict(output=str(output), files=[str(p.relative_to(ROOT)) for p in files],
                          source_bytes=sum(p.stat().st_size for p in files)), indent=2))
    if args.dry_run:
        return
    checksum = output.with_name(output.name + '.sha256')
    if output.exists() or checksum.exists():
        raise FileExistsError('Archive or checksum exists; choose a new --output')
    manifest = {str(path.relative_to(ROOT)): digest(path) for path in files}
    encoded = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for path in files:
                archive.add(path, arcname=f'project_minimal/{path.relative_to(ROOT)}', recursive=False)
            info = tarfile.TarInfo('project_minimal/SHARED_SOURCE_SHA256.json')
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
    with tarfile.open(output, 'r:gz') as archive:
        for name, expected in manifest.items():
            sha = hashlib.sha256()
            with archive.extractfile(f'project_minimal/{name}') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(block)
            if sha.hexdigest() != expected:
                raise ValueError(f'Source changed during packaging: {name}')
    checksum.write_text(f'{digest(output)}  {output.name}\n')
    print(f'Verified source archive: {output}')


if __name__ == '__main__':
    main()
