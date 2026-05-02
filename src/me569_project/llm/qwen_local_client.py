"""Qwen3.5-4B local client via mlx-vlm.

A thin adapter over the ``mlx-vlm`` library that exposes the same
``call(prompt: str) -> str`` interface as ``QwenPlusClient``, so the
Experiment 1 runner can swap between cloud and edge LLMs by changing
one line.

Model caching
-------------
The ~4.8 GB Qwen3.5-4B-MLX-8bit weights take ~4 seconds to load from the
HuggingFace cache on an M4. To avoid paying that cost on every call, the
first ``call()`` populates a set of class-level attributes
(``_cached_model``, ``_cached_processor``, ``_cached_config``,
``_cached_model_path``), and subsequent calls with the same model path
skip the load step. A new model path triggers a reload.

Thinking mode
-------------
Qwen3.5-4B is a reasoning model with ``<think>`` mode enabled by default
(per decision D006). Its replies therefore typically start with a
``<think>...</think>`` block before the actual answer, which can consume
most of a small token budget. This client defaults to ``max_tokens=2000``
to give the model room for both reasoning and a reasonably sized
Python basis function. Downstream, ``llm.code_extraction.strip_thinking_mode``
removes the reasoning block before the code is extracted, so the raw
response returned here still contains the ``<think>`` text and should
not be fed directly to ``exec``.

Threat model
------------
No network. No disk writes. Errors from ``mlx-vlm`` (OOM, missing weights,
malformed config) are wrapped in a single ``QwenLocalError`` exception so
the E1 runner can retry or count the failure the same way it does for
``QwenPlusError``.
"""
from __future__ import annotations

import time
from typing import Any

# Lazy-ish import: pull these names at module load so unittest.mock.patch
# has something to target, but do not touch disk / weights yet.
from mlx_vlm import load as _mlx_load
from mlx_vlm import generate as _mlx_generate
from mlx_vlm.prompt_utils import apply_chat_template as _mlx_apply_chat_template
from mlx_vlm.utils import load_config as _mlx_load_config


class QwenLocalError(Exception):
    """Raised for any failure in the local Qwen mlx-vlm client."""


class QwenLocalClient:
    """Local Qwen3.5-4B client using mlx-vlm on Apple Silicon.

    Parameters
    ----------
    model_path : str
        HuggingFace repo name of the MLX-converted weights. Defaults to
        ``mlx-community/Qwen3.5-4B-MLX-8bit`` per decision D006.
    max_tokens : int
        Generation token budget. Default 2000 gives room for a
        ``<think>`` preamble plus the actual function. Values below
        ~1500 typically leave the model reasoning about the task when
        the budget is exhausted, producing no usable code.
    """

    # Class-level cache so multiple client instances with the same
    # model_path share one loaded model across the process.
    _cached_model: Any = None
    _cached_processor: Any = None
    _cached_config: Any = None
    _cached_model_path: str | None = None

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3.5-4B-MLX-8bit",
        max_tokens: int = 2000,
    ) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens

        self.total_calls: int = 0
        self.last_call_time_s: float | None = None
        self.last_load_time_s: float | None = None
        self.cumulative_generation_time_s: float = 0.0

    def _ensure_loaded(self) -> None:
        """Load model weights into the class-level cache if not already loaded."""
        cls = type(self)
        if cls._cached_model_path == self.model_path and cls._cached_model is not None:
            return

        t0 = time.time()
        try:
            model, processor = _mlx_load(self.model_path)
            config = _mlx_load_config(self.model_path)
        except Exception as e:
            raise QwenLocalError(
                f"Failed to load {self.model_path}: {type(e).__name__}: {e}"
            ) from e

        cls._cached_model = model
        cls._cached_processor = processor
        cls._cached_config = config
        cls._cached_model_path = self.model_path
        self.last_load_time_s = time.time() - t0

    def call(self, prompt: str) -> str:
        """Generate a response from Qwen3.5-4B for the given user prompt.

        Returns the raw model output as a string, including any
        ``<think>...</think>`` reasoning block. Use
        ``llm.code_extraction.extract_python_code`` to strip the
        reasoning and pull out the Python source before executing.
        """
        self._ensure_loaded()
        cls = type(self)

        try:
            formatted = _mlx_apply_chat_template(
                cls._cached_processor,
                cls._cached_config,
                prompt,
                num_images=0,
            )
        except Exception as e:
            raise QwenLocalError(
                f"apply_chat_template failed: {type(e).__name__}: {e}"
            ) from e

        t0 = time.time()
        try:
            output = _mlx_generate(
                cls._cached_model,
                cls._cached_processor,
                formatted,
                max_tokens=self.max_tokens,
                verbose=False,
            )
        except Exception as e:
            raise QwenLocalError(
                f"generate failed: {type(e).__name__}: {e}"
            ) from e
        elapsed = time.time() - t0

        # mlx-vlm sometimes returns a plain string, sometimes an object
        # with a .text attribute depending on the release.
        if isinstance(output, str):
            text = output
        elif hasattr(output, "text"):
            text = str(output.text)
        else:
            text = str(output)

        self.total_calls += 1
        self.last_call_time_s = elapsed
        self.cumulative_generation_time_s += elapsed

        return text
