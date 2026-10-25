import argparse
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/checkpoints.py'
spec = importlib.util.spec_from_file_location('package_tool', MODULE)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CheckpointPackageTest(unittest.TestCase):
    def test_packaged_bytes_and_checksum_and_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = tool.ROOT
            tool.ROOT = Path(temp)
            self.addCleanup(setattr, tool, 'ROOT', previous)
            root = tool.ROOT
            checkpoint = root / 'checkpoints/cfg/experiment/100'
            (checkpoint / 'params').mkdir(parents=True)
            (checkpoint / 'params/weights').write_bytes(b'fixture weights')
            (checkpoint / '_CHECKPOINT_METADATA').write_text('{}')
            stats = root / 'assets/norm_stats.json'
            stats.parent.mkdir()
            stats.write_text('{}')
            run = root / 'logs/cfg/experiment'
            run.mkdir(parents=True)
            config = run / 'training.yaml'
            config.write_text('method: lora\n')
            metadata = dict(config='cfg', experiment='experiment', method='lora',
                checkpoint_dir='/workspace/project_minimal/checkpoints/cfg/experiment',
                norm_stats='/workspace/project_minimal/assets/norm_stats.json',
                training_yaml='/workspace/project_minimal/logs/cfg/experiment/training.yaml',
                base_params='gs://openpi-assets/checkpoints/pi05_base/params')
            (run / 'metadata.json').write_text(json.dumps(metadata))
            args = argparse.Namespace(experiment='experiment', config_name=None, step=None, dry_run=True)
            archive = tool.package(args)
            self.assertFalse(archive.exists())
            args.dry_run = False
            tool.package(args)
            self.assertEqual(tool.sha256(archive), archive.with_name(archive.name+'.sha256').read_text().split()[0])
            with tarfile.open(archive) as tar:
                self.assertEqual(tar.extractfile('checkpoint/100/params/weights').read(), b'fixture weights')
                self.assertIn('training.yaml', tar.getnames())
                self.assertIn('MANIFEST.sha256.json', tar.getnames())
            with self.assertRaises(FileExistsError):
                tool.package(args)
            archive.write_bytes(b'corrupted archive')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                tool.main(['transfer','--experiment','experiment','--destination','user@example:/tmp/output'])


if __name__ == '__main__':
    unittest.main()
