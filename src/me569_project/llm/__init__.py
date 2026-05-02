"""LLM integration layer for Experiment 1/3/4 P and Q conditions.

Provides:
- ``code_extraction``: parse LLM responses into executable Python code,
  stripping Markdown fences and Qwen-style ``<think>`` reasoning blocks.
- ``sandbox``: run extracted code in a restricted namespace with a
  pre-populated numpy / math import set and a builtins allowlist.
- (later) ``qwen_plus_client`` and ``qwen_local_client``: thin adapters
  over DashScope and mlx-vlm respectively, both exposing a uniform
  ``call(prompt: str) -> str`` interface.
"""
