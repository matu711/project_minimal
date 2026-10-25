"""Subtask language diagnostics and action conditioning on the existing π0.5 backbone."""

import dataclasses

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0


@dataclasses.dataclass(frozen=True)
class SubtaskForward:
    language_logits: jax.Array
    language_targets: jax.Array
    language_loss_mask: jax.Array
    language_ce: jax.Array
    flow_loss: jax.Array
    attention_mask: jax.Array


class Pi05Subtask(pi0.Pi0):
    """Keeps all π0.5 parameters; the tied embedder supplies vocabulary logits."""

    def embed_prefix(self, obs: _model.Observation):
        if obs.tokenized_prompt is None or obs.token_ar_mask is None or obs.token_loss_mask is None:
            raise ValueError("Subtask tokens, autoregressive mask, and CE mask are required")
        tokens, input_mask, _ = super().embed_prefix(obs)
        image_length = tokens.shape[1] - obs.tokenized_prompt.shape[1]
        image_ar = jnp.zeros((tokens.shape[0], image_length), dtype=jnp.bool_)
        ar_mask = jnp.concatenate([image_ar, jnp.asarray(obs.token_ar_mask, dtype=jnp.bool_)], axis=1)
        return tokens, input_mask, ar_mask

    def compute_loss(self, rng, observation, actions, *, train: bool = False):
        raise RuntimeError("Joint subtask/action training is reserved for Phase 3; use forward_diagnostics in Phase 2")

    def generate_subtask(self, observation: _model.Observation, *, eos_id: int = 1, max_new_tokens: int = 32):
        """Greedy batch-one generation; returns completed context for inherited action sampling.

        Phase 2 deliberately recomputes the short prefix each step. A production
        decoding cache and chunk-level scheduling belong to Phase 3.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        if observation.state.shape[0] != 1:
            raise ValueError("Phase 2 subtask generation supports batch size one")
        if bool(jnp.any(jnp.asarray(observation.token_ar_mask, dtype=jnp.bool_) & observation.tokenized_prompt_mask)):
            raise ValueError("Generation input must contain only the Task/State/Subtask prefix")
        generated = []
        for _ in range(max_new_tokens):
            text_length = int(jnp.sum(observation.tokenized_prompt_mask[0]))
            if text_length >= observation.tokenized_prompt.shape[1]:
                raise ValueError("Subtask generation exceeded max_token_len")
            prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
            attention_mask = pi0.make_attn_mask(prefix_mask, prefix_ar)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=attention_mask, positions=positions)
            image_length = prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]
            last_hidden = prefix_out[:, image_length + text_length - 1 : image_length + text_length]
            logits = self.PaliGemma.llm(last_hidden, method="decode_to_logits")
            next_id = int(jnp.argmax(logits[0, 0]))
            generated.append(next_id)
            observation = observation.replace(
                tokenized_prompt=observation.tokenized_prompt.at[0, text_length].set(next_id),
                tokenized_prompt_mask=observation.tokenized_prompt_mask.at[0, text_length].set(True),
                token_ar_mask=observation.token_ar_mask.at[0, text_length].set(True),
            )
            if next_id == eos_id:
                break
        return observation, generated

    def forward_diagnostics(
        self, rng, observation: _model.Observation, actions: _model.Actions, *, subtask_valid: jax.Array
    ) -> SubtaskForward:
        """One teacher-forced pass; returns separate CE and flow inputs without an optimizer."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=False)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        suffix_ar = jnp.broadcast_to(suffix_ar, suffix_mask.shape)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=1)
        attention_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        image_length = prefix_tokens.shape[1] - observation.tokenized_prompt.shape[1]
        text_length = observation.tokenized_prompt.shape[1]
        language_logits = self.PaliGemma.llm(
            prefix_out[:, image_length : image_length + text_length - 1], method="decode_to_logits"
        )
        targets = observation.tokenized_prompt[:, 1:]
        valid = jnp.asarray(subtask_valid, dtype=jnp.bool_)
        if valid.shape != (actions.shape[0],):
            raise ValueError(f"subtask_valid must have shape ({actions.shape[0]},), got {valid.shape}")
        loss_mask = jnp.asarray(observation.token_loss_mask[:, 1:], dtype=jnp.bool_) & valid[:, None]
        log_probs = jax.nn.log_softmax(language_logits.astype(jnp.float32), axis=-1)
        token_ce = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
        language_ce = jnp.sum(jnp.where(loss_mask, token_ce, 0.0), axis=-1) / jnp.maximum(
            jnp.sum(loss_mask, axis=-1), 1
        )
        predicted_flow = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(predicted_flow - u_t), axis=-1)
        return SubtaskForward(language_logits, targets, loss_mask, language_ce, flow_loss, attention_mask)
