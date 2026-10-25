"""TrainConfig construction adapted from the verified PiPER dual-arm trainer."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTION_FORMAT = "delta_joint_absolute_gripper"


def configure_environment(params):
    os.environ["HF_LEROBOT_HOME"] = params["data_dir"]
    os.environ["OPENPI_DATA_HOME"] = params["model_cache_dir"]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    entity = params.get("wandb_entity")
    if entity is not None:
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("wandb_entity must be a nonempty string")
        os.environ["WANDB_ENTITY"] = entity.strip()


def _boolean(params, key, default):
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def build_train_config(params):
    """Apply ALL overrides before statistics, loader construction or training."""
    configure_environment(params)  # LeRobot reads its home when first imported.
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config, optimizer, weight_loaders
    from training.adapters import bimanual_data_config, single_data_config

    if params.get("action_format", ACTION_FORMAT) != ACTION_FORMAT:
        raise ValueError(f"This trainer requires action_format={ACTION_FORMAT}")
    use_lora = _boolean(params, "use_lora", True)
    resume = _boolean(params, "resume", False)
    overwrite = _boolean(params, "overwrite", False)
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot both be true")
    repo_id = params["data_collection"]["dataset_repo_id"]
    if not repo_id.startswith("local/") or ".." in repo_id.split("/"):
        raise ValueError("Expected a local/<dataset> repo id")
    ints = {
        key: params.get(key, default)
        for key, default in {
            "action_horizon": 20 if params["robot"] == "single" else 50,
            "batch_size": 1 if params["robot"] == "single" else 8,
            "num_train_steps": 9000,
            "save_interval": 1000,
            "log_interval": 20,
            "fsdp_devices": 1,
        }.items()
    }
    if any(type(v) is not int or v <= 0 for v in ints.values()):
        raise ValueError("Horizon, batch size, steps and intervals must be positive integers")
    rank = params.get("lora_rank")
    if rank is not None and (type(rank) is not int or rank <= 0):
        raise ValueError("lora_rank must be null or a positive integer")
    workers = params.get("num_workers", 2)
    if type(workers) is not int or workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    exp_name = params.get("exp_name", f"{params['robot']}_{params['model']}_{params['method']}")
    if not isinstance(exp_name, str) or not exp_name or Path(exp_name).name != exp_name or exp_name in (".", ".."):
        raise ValueError("exp_name must be a single nonempty directory name")
    model = Pi0Config(
        pi05=params["model"] == "pi05",
        action_dim=32,
        action_horizon=ints.pop("action_horizon"),
        discrete_state_input=params["model"] == "pi05",
        paligemma_variant="gemma_2b_lora" if use_lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if use_lora else "gemma_300m",
        lora_rank=rank,
    )
    base_params = params.get("base_params_path") or f"gs://openpi-assets/checkpoints/{params['model']}_base/params"
    if "://" not in base_params:
        base_params = str(Path(base_params).resolve())
    lr = params.get("lr_schedule") or {}
    opt = params.get("optimizer") or {}
    result = config.TrainConfig(
        name=f"{params['model']}_piper_{params['robot']}_{params['method']}",
        exp_name=exp_name,
        model=model,
        data=(bimanual_data_config(repo_id, use_delta_joint_actions=True)
              if params["robot"] == "bimanual" else single_data_config(repo_id)),
        weight_loader=weight_loaders.CheckpointWeightLoader(base_params),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None if use_lora else params.get("ema_decay", 0.99),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=lr.get("warmup_steps", 1000),
            peak_lr=lr.get("peak_lr", 1e-5),
            decay_steps=lr.get("decay_steps", 10000),
            decay_lr=lr.get("decay_lr", 1e-6),
        ),
        optimizer=optimizer.AdamW(**opt),
        checkpoint_base_dir=params.get("output_dir") or str(ROOT / "checkpoints"),
        assets_base_dir=params.get("assets_dir") or str(ROOT / "assets"),
        num_workers=workers,
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=not _boolean(params, "no_wandb", False),
        project_name=params.get("wandb_project", "piper_bimanual_vla"),
        seed=params.get("seed", 42),
        **ints,
    )
    config._CONFIGS_DICT[result.name] = result
    return result

