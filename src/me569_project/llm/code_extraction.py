"""Extract executable Python code from an LLM response.

LLMs return free-form text that usually contains the requested code
wrapped in one or more Markdown code fences, often preceded by a
``<think>...</think>`` reasoning block (Qwen3.x family) or a short
natural-language preamble. This module strips all of that so the
caller gets back a clean Python source string ready for
``sandbox.run_code_in_sandbox``.

Extraction rules
----------------
1. ``<think>...</think>`` blocks are removed first. An unclosed
   ``<think>`` tag is treated as "everything after the tag is
   reasoning" and is stripped to the end of the string — this matches
   the failure mode we saw in the Week 0 mlx-vlm spike where the
   thinking budget was exhausted before the answer started.
2. All triple-backtick code fences (with or without a ``python``
   language hint) are extracted in order and concatenated with
   blank lines between them.
3. If no fences are found at all, the whole remaining response is
   treated as raw Python code (some LLMs omit the fences when the
   prompt is explicit enough).
4. An empty or whitespace-only response raises
   ``CodeExtractionError``.
"""
from __future__ import annotations

import re


class CodeExtractionError(Exception):
    """Raised when an LLM response cannot be parsed into any code."""


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)
_FENCED_CODE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+\-]*)\s*\n(.*?)```",
    re.DOTALL,
)


def strip_thinking_mode(text: str) -> str:
    """Remove Qwen-style ``<think>...</think>`` reasoning blocks.

    Handles multiple blocks in a single response as well as an unclosed
    tag (where the reasoning ran out of budget mid-sentence).
    """
    without_closed = _THINK_BLOCK_RE.sub("", text)
    without_unclosed = _UNCLOSED_THINK_RE.sub("", without_closed)
    return without_unclosed


def extract_python_code(response: str) -> str:
    """Return a single Python source string extracted from an LLM response.

    Raises
    ------
    CodeExtractionError
        If the response is empty or contains no extractable code.
    """
    if not response or not response.strip():
        raise CodeExtractionError("Empty LLM response")

    cleaned = strip_thinking_mode(response)

    matches = _FENCED_CODE_RE.findall(cleaned)
    if matches:
        code = "\n\n".join(block.strip() for block in matches).strip()
        if not code:
            raise CodeExtractionError(
                "Code fences present but contained no code"
            )
        return code

    # Fallback: treat the whole cleaned response as raw code.
    code = cleaned.strip()
    if not code:
        raise CodeExtractionError(
            "Response was empty after stripping thinking mode"
        )
    return code
