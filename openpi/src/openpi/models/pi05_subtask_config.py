"""Isolated configuration for the subtask-conditioned π0.5 path."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi05SubtaskConfig(pi0_config.Pi0Config):
    pi05: bool = True

    def __post_init__(self):
        if not self.pi05:
            raise ValueError("Pi05Subtask requires pi05=True")
        super().__post_init__()

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.pi05_subtask import Pi05Subtask

        return Pi05Subtask(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        obs, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = dataclasses.replace(
                obs,
                token_ar_mask=jax.ShapeDtypeStruct((batch_size, self.max_token_len), jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct((batch_size, self.max_token_len), jnp.bool_),
            )
        return obs, actions
