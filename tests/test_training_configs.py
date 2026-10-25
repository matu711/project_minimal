"""Validate real OpenPI model/adapter selection; no weight download or training."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('shared_train', ROOT / 'training/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class TrainingConfigTest(unittest.TestCase):
    def test_all_eight_configs_and_freeze_masks(self):
        from flax import nnx
        from openpi.models import gemma
        for robot in ('single', 'bimanual'):
            for model in ('pi0', 'pi05'):
                for method in ('lora', 'full'):
                    with self.subTest(robot=robot, model=model, method=method):
                        params = train.load_params(ROOT / 'configs' / robot / f'{model}_{method}.yaml')
                        cfg = train.build_config(params)
                        self.assertEqual(cfg.model.pi05, model == 'pi05')
                        self.assertEqual(cfg.model.max_token_len, 200 if model == 'pi05' else 48)
                        self.assertIn(f'{model}_base/params', cfg.weight_loader.params_path)
                        freeze = nnx.filterlib.to_predicate(cfg.freeze_filter)
                        base_weight = ('PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'w')
                        adapter_weight = ('PaliGemma', 'llm', 'layers', 'attn', 'q_einsum', 'lora_a')
                        self.assertEqual(freeze(base_weight, None), method == 'lora')
                        self.assertFalse(freeze(adapter_weight, None))
                        self.assertEqual(cfg.data.repo_id, params['data_collection']['dataset_repo_id'])
                        if method == 'lora':
                            self.assertEqual(gemma.get_config(cfg.model.paligemma_variant).lora_configs['attn'].rank, 16)
                            self.assertEqual(gemma.get_config(cfg.model.action_expert_variant).lora_configs['attn'].rank, 32)
                        if robot == 'bimanual':
                            self.assertEqual(cfg.data.base_config.sample_validity_key, 'quality.frame_valid')
                            self.assertTrue(cfg.data.use_delta_joint_actions)


if __name__ == '__main__':
    unittest.main()
