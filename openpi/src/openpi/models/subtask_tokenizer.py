"""PaliGemma tokenization for task/state-conditioned subtask generation."""

import dataclasses

import numpy as np
import sentencepiece

import openpi.shared.download as download


@dataclasses.dataclass(frozen=True)
class SubtaskTokens:
    ids: np.ndarray
    input_mask: np.ndarray
    ar_mask: np.ndarray
    loss_mask: np.ndarray
    prefix_length: int


class SubtaskTokenizer:
    """Uses the π0.5 state bins and a single train/inference prefix encoding."""

    def __init__(self, max_len: int = 200):
        self.max_len = max_len
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as model_file:
            self._sp = sentencepiece.SentencePieceProcessor(model_proto=model_file.read())

    @staticmethod
    def prefix_text(prompt: str, state: np.ndarray) -> str:
        state = np.asarray(state)
        if state.shape != (8,):
            raise ValueError(f"Expected eight normalized state values, got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("Normalized state contains non-finite values")
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if not cleaned:
            raise ValueError("High-level task is empty")
        bins = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_text = " ".join(map(str, bins))
        return f"Task: {cleaned}\nState: {state_text}\nSubtask: "

    def tokenize(
        self, prompt: str, state: np.ndarray, *, subtask: str | None = None, subtask_valid: bool | None = None
    ) -> SubtaskTokens:
        if (subtask is None) != (subtask_valid is None):
            raise ValueError("subtask and subtask_valid must be supplied together")
        prefix = self._sp.encode(self.prefix_text(prompt, state), add_bos=True)
        if subtask is None:
            suffix = []
            valid = False
        else:
            valid = bool(subtask_valid)
            if valid and not subtask.strip():
                raise ValueError("Valid subtask text is empty")
            if not valid and subtask.strip():
                raise ValueError("Invalid subtask must have empty text")
            suffix = self._sp.encode(subtask.strip(), add_eos=True) if valid else [self._sp.eos_id()]
        ids = prefix + suffix
        if len(ids) > self.max_len:
            raise ValueError(f"Subtask token sequence {len(ids)} exceeds max_len={self.max_len}")
        pad = self.max_len - len(ids)
        return SubtaskTokens(
            ids=np.asarray(ids + [0] * pad, dtype=np.int32),
            input_mask=np.asarray([True] * len(ids) + [False] * pad, dtype=np.bool_),
            ar_mask=np.asarray([0] * len(prefix) + [1] * len(suffix) + [0] * pad, dtype=np.int32),
            loss_mask=np.asarray([False] * len(prefix) + [valid] * len(suffix) + [False] * pad, dtype=np.bool_),
            prefix_length=len(prefix),
        )
