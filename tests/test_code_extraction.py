"""Tests for LLM response -> Python code extraction."""
import pytest

from me569_project.llm.code_extraction import (
    CodeExtractionError,
    extract_python_code,
    strip_thinking_mode,
)


def test_extract_python_fenced_block():
    response = """
Here is the function:

```python
def basis(x):
    return [1.0, x[0]]
```

Hope this helps!
"""
    code = extract_python_code(response)
    assert code == "def basis(x):\n    return [1.0, x[0]]"


def test_extract_generic_fenced_block():
    response = "```\ndef basis(x):\n    return [x[0]]\n```"
    code = extract_python_code(response)
    assert code == "def basis(x):\n    return [x[0]]"


def test_extract_multiple_fenced_blocks_concatenated():
    response = """
First block:

```python
import numpy as np
```

Second block:

```python
def basis(x):
    return [np.sin(x[2])]
```
"""
    code = extract_python_code(response)
    assert "import numpy as np" in code
    assert "def basis(x):" in code


def test_extract_raw_python_without_fences():
    """When no markdown fences are present, the entire response is treated as code."""
    response = "def basis(x):\n    return [1.0]\n"
    code = extract_python_code(response)
    assert code == "def basis(x):\n    return [1.0]"


def test_strip_thinking_mode_before_extract():
    response = """<think>
Let me analyze the request carefully.
Step 1: Figure out the function signature.
Step 2: Write the function.
</think>

```python
def basis(x):
    return [1.0, x[0], x[1]]
```
"""
    code = extract_python_code(response)
    assert code == "def basis(x):\n    return [1.0, x[0], x[1]]"
    assert "Step 1" not in code


def test_strip_thinking_mode_standalone_helper():
    """The thinking-strip helper is callable standalone and handles multiple tags."""
    text = "<think>blah</think>visible1<think>more</think>visible2"
    assert strip_thinking_mode(text) == "visible1visible2"


def test_strip_thinking_mode_handles_no_tags():
    text = "no thinking here"
    assert strip_thinking_mode(text) == "no thinking here"


def test_strip_thinking_mode_unclosed_tag_is_stripped():
    """Qwen sometimes opens a <think> without closing it; we strip to end."""
    text = "<think>runaway reasoning without closing"
    assert strip_thinking_mode(text) == ""


def test_extract_from_qwen_style_thinking_response():
    """Realistic Qwen response with thinking prefix and fenced code."""
    response = """<think>
Let me work through this step by step.
The function needs to return a list of basis functions.
</think>

Here's the basis function:

```python
import numpy as np

def basis(x):
    return [
        1.0,
        x[0],
        x[1],
        np.sin(x[2]),
        np.cos(x[2]),
    ]
```
"""
    code = extract_python_code(response)
    assert "def basis(x):" in code
    assert "np.sin(x[2])" in code
    assert "Let me work" not in code


def test_extract_empty_response_raises():
    with pytest.raises(CodeExtractionError):
        extract_python_code("")


def test_extract_whitespace_only_raises():
    with pytest.raises(CodeExtractionError):
        extract_python_code("    \n\n   \n")
