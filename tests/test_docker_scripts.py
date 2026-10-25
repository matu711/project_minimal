"""Exercise shell lifecycle through a fake Docker CLI (never touches Docker)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DockerScriptsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tuning scripts ')
        self.root = Path(self.tmp.name) / 'project with spaces'
        (self.root / 'scripts').mkdir(parents=True)
        for filename in ('build.sh', 'run.sh', 'scripts/docker-common.sh'):
            shutil.copy(ROOT / filename, self.root / filename)
        self.bin = Path(self.tmp.name) / 'bin'
        self.bin.mkdir()
        self.log = Path(self.tmp.name) / 'calls.jsonl'
        docker = self.bin / 'docker'
        docker.write_text('''#!/usr/bin/env python3
import json,os,sys
a=sys.argv[1:]
with open(os.environ['CALL_LOG'],'a') as f: f.write(json.dumps(a)+'\\n')
if a[:2]==['image','inspect']: sys.exit(int(os.environ.get('IMAGE_MISSING','0')))
if a[:1]==['inspect']: print(os.environ.get('CONTAINER_STATE','running'))
if 'ps' in a and os.environ.get('CONTAINER_EXISTS')=='1': print('new-tuning-id')
if 'build' in a: sys.exit(int(os.environ.get('BUILD_FAILURE','0')))
''')
        docker.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin)+os.pathsep+os.environ['PATH'], CALL_LOG=str(self.log),
                        DISPLAY='', WAYLAND_DISPLAY='')

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self, filename='run.sh', **environment):
        self.env.update(environment)
        result = subprocess.run(['sh', str(self.root / filename)], cwd='/tmp', env=self.env,
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return result, calls

    def test_missing_image_guides_build_without_start(self):
        result, calls = self.execute(IMAGE_MISSING='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('build.sh', result.stderr)
        self.assertFalse(any('up' in c or 'build' in c or 'exec' in c for c in calls))

    def test_running_container_reused_even_without_image(self):
        result, calls = self.execute(CONTAINER_EXISTS='1', IMAGE_MISSING='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('up' in c or 'start' in c or 'build' in c for c in calls))
        self.assertIn('--workdir', calls[-1])
        self.assertEqual(calls[-1][-1], 'sh')

    def test_terminator_launch_preserves_path_with_spaces(self):
        terminal_log = Path(self.tmp.name) / 'terminal.json'
        terminal = self.bin / 'terminator'
        terminal.write_text('''#!/usr/bin/env python3
import json,os,sys
with open(os.environ['TERMINAL_LOG'],'w') as f: json.dump(sys.argv[1:],f)
''')
        terminal.chmod(0o755)
        result, calls = self.execute(CONTAINER_EXISTS='1', DISPLAY=':0', TERMINAL_LOG=str(terminal_log))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(terminal_log.read_text()),
                         ['--working-directory', str(self.root), '-x', 'sh', str(self.root/'run.sh'), '--attach'])
        self.assertFalse(any('exec' in c for c in calls))

    def test_stopped_container_started_without_recreation(self):
        result, calls = self.execute(CONTAINER_EXISTS='1', CONTAINER_STATE='exited')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(any('start' in c for c in calls))
        self.assertFalse(any('up' in c or 'build' in c for c in calls))

    def test_new_container_never_builds_or_pulls(self):
        result, calls = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)
        up = next(c for c in calls if 'up' in c)
        self.assertIn('--no-build', up)
        self.assertEqual(up[up.index('--pull')+1], 'never')
        self.assertIn(str(self.root), up)
        self.assertTrue((self.root / 'exports').is_dir())

    def test_build_error_propagates(self):
        result, calls = self.execute('build.sh', BUILD_FAILURE='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ビルドに失敗', result.stderr)
        self.assertTrue(any(c[-2:] == ['build', 'tuning'] for c in calls))


if __name__ == '__main__':
    unittest.main()
