# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.rest.anthropic import RestAnthropicLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="claude-opus-4-6") -> RestAnthropicLlm:
    with patch.object(RestAnthropicLlm, "_create_client", return_value=MagicMock()):
        return RestAnthropicLlm(api_key="fake-key", model_name=model_name)


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[LlmMessage(role=Role.user, content="Hello")])
    return LlmRequest(**(defaults | kwargs))


def _api_response(content="Hi there", model="claude-opus-4-6", stop_reason="end_turn",
                  input_tokens=10, output_tokens=5) -> dict:
    return {
        "content": [{"text": content}],
        "model": model,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


class TestRestAnthropicLlmInit:
    def test_model_provider_set(self):
        assert _llm().model_provider == "anthropic"

    def test_base_url(self):
        assert RestAnthropicLlm.ANTHROPIC_BASE_URL == "https://api.anthropic.com/v1"

    def test_anthropic_version(self):
        assert RestAnthropicLlm.ANTHROPIC_VERSION == "2023-06-01"


class TestRestAnthropicConvertMessages:
    def test_user_message_included(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.user, content="Hello")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_message_excluded(self):
        llm = _llm()
        msgs = [
            LlmMessage(role=Role.system, content="You are helpful."),
            LlmMessage(role=Role.user, content="Hello"),
        ]
        result = llm._convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_assistant_message_included(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.assistant, content="Hi")]
        result = llm._convert_messages(msgs)
        assert result[0]["role"] == "assistant"


class TestRestAnthropicBuildBody:
    def test_includes_model(self):
        llm = _llm("claude-opus-4-6")
        body = llm._build_body(_request())
        assert body["model"] == "claude-opus-4-6"

    def test_max_tokens_defaults_to_4096(self):
        llm = _llm()
        body = llm._build_body(_request(max_tokens=None))
        assert body["max_tokens"] == 4096

    def test_max_tokens_used_when_set(self):
        llm = _llm()
        body = llm._build_body(_request(max_tokens=1024))
        assert body["max_tokens"] == 1024

    def test_system_prompt_included_when_set(self):
        llm = _llm()
        body = llm._build_body(_request(system_prompt="Be concise."))
        assert body["system"] == "Be concise."

    def test_system_prompt_absent_when_not_set(self):
        llm = _llm()
        body = llm._build_body(_request())
        assert "system" not in body

    def test_temperature_included(self):
        llm = _llm()
        body = llm._build_body(_request(temperature=0.5))
        assert body["temperature"] == 0.5


class TestRestAnthropicGenerate:
    def test_returns_llm_response(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response()
        llm._client.post = AsyncMock(return_value=mock_resp)

        response = asyncio.run(llm._generate(_request()))
        assert isinstance(response, LlmResponse)

    def test_content_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(content="Answer")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).content == "Answer"

    def test_model_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(model="claude-opus-4-6")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).model == "claude-opus-4-6"

    def test_finish_reason_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(stop_reason="end_turn")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).finish_reason == "end_turn"

    def test_usage_totalled(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(input_tokens=10, output_tokens=5)
        llm._client.post = AsyncMock(return_value=mock_resp)

        usage = asyncio.run(llm._generate(_request())).usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15


class TestRestAnthropicStream:
    def test_yields_content_block_delta_text(self):
        llm = _llm()
        lines = [
            'data: ' + json.dumps({"type": "content_block_delta", "delta": {"text": "Hello"}}),
            'data: ' + json.dumps({"type": "content_block_delta", "delta": {"text": " world"}}),
            "data: [DONE]",
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _aiter(*lines)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        llm._client.stream.return_value = mock_cm

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hello", " world"]

    def test_non_data_lines_skipped(self):
        llm = _llm()
        lines = [
            "event: message_start",
            'data: ' + json.dumps({"type": "content_block_delta", "delta": {"text": "Hi"}}),
            "data: [DONE]",
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _aiter(*lines)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        llm._client.stream.return_value = mock_cm

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hi"]

    def test_non_content_block_events_skipped(self):
        llm = _llm()
        lines = [
            'data: ' + json.dumps({"type": "message_start"}),
            'data: ' + json.dumps({"type": "content_block_delta", "delta": {"text": "Hi"}}),
            "data: [DONE]",
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _aiter(*lines)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        llm._client.stream.return_value = mock_cm

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hi"]


class TestRestAnthropicListModels:
    def test_returns_sorted_model_ids(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "claude-sonnet-4-6"}, {"id": "claude-opus-4-6"}]}
        llm._client.get = AsyncMock(return_value=mock_resp)

        models = asyncio.run(llm.list_models())
        assert models == sorted(["claude-sonnet-4-6", "claude-opus-4-6"])
