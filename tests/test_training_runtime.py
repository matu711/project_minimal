"""Exercise statistics, normalized data, and real Orbax save/restore on small CPU fixtures."""
import dataclasses
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runtime_train_tests', ROOT/'training/train.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


def test_normalization_and_batch_roundtrip(tmp_path, monkeypatch):
    from conversion.pipeline import convert
    from tests.conversion_fixtures import record
    source = tmp_path/'raw'
    record(source)
    dataset = tmp_path/'datasets/local/stats_test'
    convert('bimanual', [source], dataset, 'local/stats_test')
    params = train.load_params(ROOT/'configs/bimanual/pi05_lora.yaml')
    params.update(data_dir=str(tmp_path/'datasets'), assets_dir=str(tmp_path/'assets'),
                  output_dir=str(tmp_path/'checkpoints'), batch_size=2, num_workers=0, action_horizon=3)
    params['data_collection']['dataset_repo_id'] = 'local/stats_test'
    params['data_collection']['save_dir'] = params['data_dir']
    cfg = train.build_config(params)
    # Conversion imported LeRobot before this test selected its temporary data root.
    # Normal CLI invocations configure the environment before importing LeRobot.
    from lerobot.common.datasets import lerobot_dataset
    monkeypatch.setattr(lerobot_dataset, 'HF_LEROBOT_HOME', Path(params['data_dir']))
    stats = train.ensure_norm_stats(cfg, params)
    assert stats.is_file()
    batch = train.validate_batch(cfg, 'bimanual')
    assert batch['normalization_checked']
    assert batch['absolute_action_roundtrip_verified']
    assert batch['action_shape'] == [2,3,32]
    resume = dataclasses.replace(cfg, resume=True)
    assert train.ensure_norm_stats(resume, params) == stats
    import pytest
    with pytest.raises(ValueError, match='Cannot recompute'):
        train.ensure_norm_stats(resume, params, recompute=True)
    different = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_horizon=4))
    with pytest.raises(ValueError, match='configuration changed'):
        train.ensure_norm_stats(different, params)


def test_real_orbax_checkpoint_save_and_resume(tmp_path):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax
    from openpi.training import checkpoints, utils
    from openpi.shared import array_typing as at

    graph, params = nnx.split(nnx.Linear(2, 1, rngs=nnx.Rngs(0)))
    tx = optax.adam(1e-3)
    gradients = jax.tree.map(jnp.ones_like, params)
    updates, opt_state = tx.update(gradients, tx.init(params))
    trained_params = optax.apply_updates(params, updates)
    with at.disable_typechecking():
        state = utils.TrainState(step=jnp.array(5), params=trained_params, model_def=graph,
                                 opt_state=opt_state, tx=tx, ema_decay=None)

    class Loader:
        def data_config(self):
            from openpi.training.config import DataConfig
            return DataConfig()

    manager, resuming = checkpoints.initialize_checkpoint_dir(tmp_path/'run', keep_period=None,
                                                              overwrite=False, resume=False)
    assert not resuming
    checkpoints.save_state(manager, state, Loader(), 5)
    manager.wait_until_finished()
    manager.close()
    assert (tmp_path/'run/5/_CHECKPOINT_METADATA').is_file()
    manager, resuming = checkpoints.initialize_checkpoint_dir(tmp_path/'run', keep_period=None,
                                                              overwrite=False, resume=True)
    assert resuming
    template = dataclasses.replace(state, step=jnp.array(0), params=jax.tree.map(jnp.zeros_like, params))
    restored = checkpoints.restore_state(manager, template, Loader())
    manager.close()
    assert int(restored.step) == 5
    for expected, actual in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored.params), strict=True):
        np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(jax.tree.leaves(state.opt_state), jax.tree.leaves(restored.opt_state), strict=True):
        np.testing.assert_array_equal(actual, expected)
