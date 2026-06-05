"""Qwen-Plus (Alibaba DashScope) client wrapper.

A thin adapter over ``dashscope.Generation.call`` that exposes a uniform
``call(prompt: str) -> str`` interface for the E1/E3/E4 LLM conditions.
Both this client and the planned ``QwenLocalClient`` (mlx-vlm backend)
implement the same method signature, so the experiment runner can swap
between them by changing one line.

Responsibilities
----------------
- Read the DashScope API key from the constructor argument or the
  ``DASHSCOPE_API_KEY`` environment variable.
- Build a ``messages``-format request targeting the ``qwen-plus`` alias
  (which, as of April 2026, is backed by Qwen 3.6).
- Translate HTTP errors, rate limits, and missing output into a single
  ``QwenPlusError`` exception type.
- Track per-call and cumulative usage (tokens, calls) for cost logging
  in the experiment runner.
"""
from __future__ import annotations

import os
from typing import Any

from dashscope import Generation


class QwenPlusError(Exception):
    """Raised for any failure when calling the Qwen-Plus API."""


class QwenPlusClient:
    """Stateful Qwen-Plus DashScope client.

    Parameters
    ----------
    api_key : str, optional
        DashScope API key. If omitted, read from the
        ``DASHSCOPE_API_KEY`` environment variable.
    model : str
        DashScope model alias. Default ``qwen-plus`` routes to
        Alibaba's current commercial frontier (Qwen 3.6 as of
        April 2026).
    temperature : float
        Sampling temperature. Default 0.3 favors deterministic code
        generation.
    max_tokens : int
        Maximum response tokens. Default 1500 accommodates a short
        reasoning preamble plus a 15–30 line Python basis function.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen-plus",
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "api_key is required (pass explicitly or set "
                "DASHSCOPE_API_KEY in the environment)"
            )

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.total_calls: int = 0
        self.cumulative_tokens: int = 0
        self.last_usage: dict[str, Any] | None = None
        self.last_request_id: str | None = None

    def call(self, prompt: str) -> str:
        """Send a single user prompt to Qwen-Plus and return the reply text.

        Raises
        ------
        QwenPlusError
            On any non-200 status, missing output, or malformed response.
        """
        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            code = getattr(response, "code", "unknown")
            message = getattr(response, "message", "no message")
            raise QwenPlusError(
                f"DashScope call failed: status={status_code} "
                f"code={code} message={message}"
            )

        # Pull the text out of the message-format response
        try:
            text = response.output.choices[0].message.content
        except (AttributeError, IndexError, KeyError) as e:
            raise QwenPlusError(
                f"Unexpected response structure: {e}"
            ) from e

        if text is None:
            raise QwenPlusError("Response text was None")

        # Bookkeeping for cost tracking
        self.total_calls += 1
        self.last_request_id = getattr(response, "request_id", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = dict(usage)
            self.cumulative_tokens += int(
                self.last_usage.get("total_tokens", 0)
            )

        return text
