# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.sdk.anthropic import NativeSdkAnthropicLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="claude-opus-4-6") -> NativeSdkAnthropicLlm:
    with patch.object(NativeSdkAnthropicLlm, "_create_client", return_value=MagicMock()):
        return NativeSdkAnthropicLlm(api_key="fake-key", model_name=model_name)


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[LlmMessage(role=Role.user, content="Hello")])
    return LlmRequest(**(defaults | kwargs))


def _mock_generate_response(content="Hi there", model="claude-opus-4-6", stop_reason="end_turn"):
    resp = MagicMock()
    resp.content = [MagicMock(text=content)]
    resp.model = model
    resp.stop_reason = stop_reason
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    return resp


class TestNativeSdkAnthropicLlmInit:
    def test_model_provider_set(self):
        llm = _llm()
        assert llm.model_provider == "anthropic"

    def test_model_name_set(self):
        llm = _llm("claude-opus-4-6")
        assert llm.model_name == "claude-opus-4-6"

    def test_model_name_none(self):
        llm = _llm(model_name=None)
        assert llm.model_name is None


class TestSdkAnthropicConvertMessages:
    def test_user_message_included(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.user, content="Hello")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_included(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.assistant, content="Hi")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi"}]

    def test_system_message_excluded(self):
        llm = _llm()
        msgs = [
            LlmMessage(role=Role.system, content="You are helpful."),
            LlmMessage(role=Role.user, content="Hello"),
        ]
        result = llm._convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_messages(self):
        llm = _llm()
        assert llm._convert_messages([]) == []

    def test_mixed_messages_order_preserved(self):
        llm = _llm()
        msgs = [
            LlmMessage(role=Role.user, content="First"),
            LlmMessage(role=Role.assistant, content="Second"),
            LlmMessage(role=Role.user, content="Third"),
        ]
        result = llm._convert_messages(msgs)
        assert [r["content"] for r in result] == ["First", "Second", "Third"]


class TestSdkAnthropicBuildKwargs:
    def test_includes_messages(self):
        llm = _llm()
        request = _request()
        kwargs = llm._build_kwargs(request)
        assert "messages" in kwargs

    def test_system_prompt_included_when_set(self):
        llm = _llm()
        request = _request(system_prompt="Be concise.")
        kwargs = llm._build_kwargs(request)
        assert kwargs["system"] == "Be concise."

    def test_system_prompt_absent_when_not_set(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request())
        assert "system" not in kwargs

    def test_max_tokens_defaults_to_zero_when_none(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request(max_tokens=None))
        assert kwargs["max_tokens"] == 0

    def test_max_tokens_used_when_set(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request(max_tokens=512))
        assert kwargs["max_tokens"] == 512


class TestSdkAnthropicGenerate:
    def test_returns_llm_response(self):
        llm = _llm()
        llm._client.messages.create = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert isinstance(response, LlmResponse)

    def test_content_mapped(self):
        llm = _llm()
        llm._client.messages.create = AsyncMock(return_value=_mock_generate_response(content="Answer"))
        response = asyncio.run(llm._generate(_request()))
        assert response.content == "Answer"

    def test_model_mapped(self):
        llm = _llm()
        llm._client.messages.create = AsyncMock(return_value=_mock_generate_response(model="claude-opus-4-6"))
        response = asyncio.run(llm._generate(_request()))
        assert response.model == "claude-opus-4-6"

    def test_finish_reason_mapped(self):
        llm = _llm()
        llm._client.messages.create = AsyncMock(return_value=_mock_generate_response(stop_reason="end_turn"))
        response = asyncio.run(llm._generate(_request()))
        assert response.finish_reason == "end_turn"

    def test_usage_mapped(self):
        llm = _llm()
        llm._client.messages.create = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15


class TestSdkAnthropicStream:
    def test_yields_text_chunks(self):
        llm = _llm()
        mock_stream_obj = MagicMock()
        mock_stream_obj.text_stream = _aiter("Hello", " world")
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_obj)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        llm._client.messages.stream.return_value = mock_cm

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hello", " world"]

    def test_stream_via_generate(self):
        llm = _llm()
        mock_stream_obj = MagicMock()
        mock_stream_obj.text_stream = _aiter("Hi")
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_stream_obj)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        llm._client.messages.stream.return_value = mock_cm

        async def _run():
            gen = await llm.generate(_request(), stream=True)
            return [c async for c in gen]

        chunks = asyncio.run(_run())
        assert chunks == ["Hi"]


class TestSdkAnthropicListModels:
    def test_returns_model_ids(self):
        llm = _llm()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"id": "claude-opus-4-6"}, {"id": "claude-sonnet-4-6"}]}

        mock_http_client = MagicMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)
        mock_http_client.get = AsyncMock(return_value=mock_response)

        with patch("relay.llm.native.sdk.anthropic.httpx.AsyncClient", return_value=mock_http_client):
            models = asyncio.run(llm.list_models())

        assert "claude-opus-4-6" in models
        assert "claude-sonnet-4-6" in models
