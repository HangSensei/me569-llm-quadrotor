"""Tests for the Qwen-Plus DashScope client wrapper.

These tests mock out dashscope.Generation.call so they run with no real
network calls and no API key. A separate integration script is used
to verify against the real API outside of the pytest suite.
"""
from unittest import mock

import pytest

from me569_project.llm.qwen_plus_client import (
    QwenPlusClient,
    QwenPlusError,
)


def _fake_successful_response(text: str):
    """Build a dashscope-shaped response object with the given message text."""
    resp = mock.Mock()
    resp.status_code = 200
    resp.request_id = "req-fake-123"
    resp.output = mock.Mock()
    resp.output.choices = [mock.Mock()]
    resp.output.choices[0].message = mock.Mock()
    resp.output.choices[0].message.content = text
    resp.usage = {
        "input_tokens": 50,
        "output_tokens": 25,
        "total_tokens": 75,
    }
    return resp


def _fake_failed_response(status_code: int, code: str, message: str):
    resp = mock.Mock()
    resp.status_code = status_code
    resp.code = code
    resp.message = message
    resp.request_id = "req-fake-err"
    return resp


@pytest.fixture
def client():
    return QwenPlusClient(api_key="sk-fake-test-key", max_tokens=1500)


def test_client_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        QwenPlusClient(api_key="")


def test_client_uses_env_var_when_key_not_provided(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env-test")
    c = QwenPlusClient()
    assert c.api_key == "sk-env-test"


def test_client_raises_if_no_key_anywhere(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="api_key"):
        QwenPlusClient()


def test_call_returns_text_on_success(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_successful_response("```python\ndef basis(xu):\n    return [1.0]\n```"),
    ):
        text = client.call("write a basis function")

    assert "def basis" in text
    assert "```python" in text


def test_call_records_last_usage(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_successful_response("hi"),
    ):
        client.call("hello")

    assert client.last_usage == {
        "input_tokens": 50,
        "output_tokens": 25,
        "total_tokens": 75,
    }
    assert client.last_request_id == "req-fake-123"


def test_call_raises_on_http_error(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_failed_response(401, "InvalidApiKey", "Bad key"),
    ):
        with pytest.raises(QwenPlusError, match="401"):
            client.call("anything")


def test_call_raises_on_rate_limit(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_failed_response(429, "Throttling", "rate limit"),
    ):
        with pytest.raises(QwenPlusError, match="429"):
            client.call("anything")


def test_call_passes_user_message_correctly(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_successful_response("ok"),
    ) as m:
        client.call("my prompt")

    kwargs = m.call_args.kwargs
    assert kwargs["model"] == "qwen-plus"
    assert kwargs["messages"] == [{"role": "user", "content": "my prompt"}]
    assert kwargs["temperature"] == pytest.approx(0.3)
    assert kwargs["max_tokens"] == 1500


def test_call_accepts_custom_temperature_and_max_tokens():
    c = QwenPlusClient(api_key="sk-fake", temperature=0.7, max_tokens=500)
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_successful_response("ok"),
    ) as m:
        c.call("p")

    assert m.call_args.kwargs["temperature"] == pytest.approx(0.7)
    assert m.call_args.kwargs["max_tokens"] == 500


def test_total_calls_counter(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_successful_response("a"),
    ):
        client.call("1")
        client.call("2")
        client.call("3")

    assert client.total_calls == 3


def test_client_tracks_cumulative_tokens(client):
    with mock.patch(
        "me569_project.llm.qwen_plus_client.Generation.call",
        return_value=_fake_successful_response("a"),
    ):
        client.call("1")
        client.call("2")

    # Two calls × 75 total tokens each = 150 total
    assert client.cumulative_tokens == 150
