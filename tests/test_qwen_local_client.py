"""Tests for the Qwen3.5-4B local mlx-vlm client.

These tests mock out ``mlx_vlm.load``, ``mlx_vlm.generate``,
``apply_chat_template``, and ``load_config`` so the suite runs without
loading the 4.8 GB model weights. A separate integration script
exercises the real model outside pytest.
"""
from unittest import mock

import pytest

from me569_project.llm import qwen_local_client as qlc_module
from me569_project.llm.qwen_local_client import QwenLocalClient, QwenLocalError


@pytest.fixture(autouse=True)
def reset_singleton_cache():
    """Clear the module-level model cache before and after each test."""
    QwenLocalClient._cached_model = None
    QwenLocalClient._cached_processor = None
    QwenLocalClient._cached_config = None
    QwenLocalClient._cached_model_path = None
    yield
    QwenLocalClient._cached_model = None
    QwenLocalClient._cached_processor = None
    QwenLocalClient._cached_config = None
    QwenLocalClient._cached_model_path = None


def test_client_initializes_without_loading_the_model():
    """Constructing the client must not touch disk or import a 4.8 GB model."""
    with mock.patch.object(qlc_module, "_mlx_load") as load_mock:
        QwenLocalClient()
    load_mock.assert_not_called()


def test_first_call_triggers_model_load():
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())) as load_mock, \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()) as config_mock, \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="formatted prompt") as tpl_mock, \
         mock.patch.object(qlc_module, "_mlx_generate", return_value="def basis(xu): return [1.0]") as gen_mock:
        client = QwenLocalClient()
        result = client.call("write a basis function")

    assert result == "def basis(xu): return [1.0]"
    load_mock.assert_called_once_with("mlx-community/Qwen3.5-4B-MLX-8bit")
    config_mock.assert_called_once_with("mlx-community/Qwen3.5-4B-MLX-8bit")
    tpl_mock.assert_called_once()
    gen_mock.assert_called_once()


def test_subsequent_calls_reuse_cached_model():
    """The second call must not reload the model weights."""
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())) as load_mock, \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="p"), \
         mock.patch.object(qlc_module, "_mlx_generate", return_value="out"):
        client = QwenLocalClient()
        client.call("prompt 1")
        client.call("prompt 2")
        client.call("prompt 3")

    assert load_mock.call_count == 1


def test_different_model_paths_reload_independently():
    """Constructing two clients with different model paths should trigger two loads."""
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())) as load_mock, \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="p"), \
         mock.patch.object(qlc_module, "_mlx_generate", return_value="out"):
        client_a = QwenLocalClient(model_path="path/A")
        client_a.call("p")
        client_b = QwenLocalClient(model_path="path/B")
        client_b.call("p")

    assert load_mock.call_count == 2


def test_call_passes_max_tokens_to_generate():
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())), \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="p"), \
         mock.patch.object(qlc_module, "_mlx_generate", return_value="out") as gen_mock:
        client = QwenLocalClient(max_tokens=3000)
        client.call("test")

    assert gen_mock.call_args.kwargs.get("max_tokens") == 3000


def test_call_tracks_call_count_and_latency():
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())), \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="p"), \
         mock.patch.object(qlc_module, "_mlx_generate", return_value="out"):
        client = QwenLocalClient()
        client.call("a")
        client.call("b")

    assert client.total_calls == 2
    assert client.last_call_time_s is not None
    assert client.last_call_time_s >= 0.0


def test_call_accepts_object_output_with_text_attribute():
    """Some mlx-vlm versions return an object with a .text attribute instead of a plain string."""
    class FakeOutput:
        def __init__(self, text):
            self.text = text

    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())), \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="p"), \
         mock.patch.object(qlc_module, "_mlx_generate", return_value=FakeOutput("code here")):
        client = QwenLocalClient()
        result = client.call("test")

    assert result == "code here"


def test_call_raises_on_generate_failure():
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())), \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", return_value="p"), \
         mock.patch.object(qlc_module, "_mlx_generate", side_effect=RuntimeError("Metal OOM")):
        client = QwenLocalClient()
        with pytest.raises(QwenLocalError, match="Metal OOM"):
            client.call("test")


def test_call_raises_on_load_failure():
    with mock.patch.object(qlc_module, "_mlx_load", side_effect=OSError("missing model")):
        client = QwenLocalClient()
        with pytest.raises(QwenLocalError, match="missing model"):
            client.call("test")


def test_call_raises_on_apply_chat_template_failure():
    with mock.patch.object(qlc_module, "_mlx_load", return_value=(mock.Mock(), mock.Mock())), \
         mock.patch.object(qlc_module, "_mlx_load_config", return_value=mock.Mock()), \
         mock.patch.object(qlc_module, "_mlx_apply_chat_template", side_effect=ValueError("bad template")):
        client = QwenLocalClient()
        with pytest.raises(QwenLocalError, match="bad template"):
            client.call("test")


def test_default_model_path_is_qwen3_5_4b_mlx_8bit():
    client = QwenLocalClient()
    assert client.model_path == "mlx-community/Qwen3.5-4B-MLX-8bit"


def test_default_max_tokens_is_at_least_1500():
    """Thinking-mode budget must accommodate a reasoning preamble plus code."""
    client = QwenLocalClient()
    assert client.max_tokens >= 1500
