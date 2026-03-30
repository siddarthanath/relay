# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.rest.openai import NativeRestOpenAILlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="gpt-4o") -> NativeRestOpenAILlm:
    with patch.object(NativeRestOpenAILlm, "_create_client", return_value=MagicMock()):
        return NativeRestOpenAILlm(api_key="fake-key", model_name=model_name)


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[LlmMessage(role=Role.user, content="Hello")])
    return LlmRequest(**(defaults | kwargs))


def _api_response(content="Hi there", model="gpt-4o", finish_reason="stop",
                  prompt_tokens=10, completion_tokens=5, total_tokens=15) -> dict:
    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


class TestNativeRestOpenAILlmInit:
    def test_model_provider_set(self):
        assert _llm().model_provider == "openai"

    def test_base_url(self):
        assert NativeRestOpenAILlm.OPENAI_BASE_URL == "https://api.openai.com/v1"


class TestRestOpenAIConvertMessages:
    def test_user_message_converted(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.user, content="Hello")]
        assert llm._convert_messages(msgs) == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_converted(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.assistant, content="Hi")]
        assert llm._convert_messages(msgs) == [{"role": "assistant", "content": "Hi"}]

    def test_system_message_included(self):
        # OpenAI accepts system role natively
        llm = _llm()
        msgs = [LlmMessage(role=Role.system, content="Be helpful.")]
        result = llm._convert_messages(msgs)
        assert result[0]["role"] == "system"

    def test_empty_messages(self):
        assert _llm()._convert_messages([]) == []


class TestRestOpenAIBuildMessages:
    def test_system_prompt_included_when_set(self):
        llm = _llm()
        msgs = llm._build_messages(_request(system_prompt="You are helpful."))
        assert msgs[0] == {"role": "system", "content": "You are helpful."}

    def test_system_prompt_absent_when_not_set(self):
        llm = _llm()
        msgs = llm._build_messages(_request(system_prompt=None))
        assert not any(m["role"] == "system" for m in msgs)


class TestRestOpenAIBuildBody:
    def test_model_included(self):
        llm = _llm("gpt-4o")
        body = llm._build_body(_request())
        assert body["model"] == "gpt-4o"

    def test_temperature_included(self):
        llm = _llm()
        body = llm._build_body(_request(temperature=0.5))
        assert body["temperature"] == 0.5

    def test_top_p_included_when_set(self):
        llm = _llm()
        body = llm._build_body(_request(top_p=0.9))
        assert body["top_p"] == 0.9

    def test_top_p_absent_when_not_set(self):
        llm = _llm()
        body = llm._build_body(_request())
        assert "top_p" not in body

    def test_frequency_penalty_included_when_set(self):
        llm = _llm()
        body = llm._build_body(_request(frequency_penalty=0.5))
        assert body["frequency_penalty"] == 0.5

    def test_presence_penalty_included_when_set(self):
        llm = _llm()
        body = llm._build_body(_request(presence_penalty=0.3))
        assert body["presence_penalty"] == 0.3

    def test_frequency_penalty_absent_when_not_set(self):
        llm = _llm()
        body = llm._build_body(_request())
        assert "frequency_penalty" not in body


class TestRestOpenAIGenerate:
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
        mock_resp.json.return_value = _api_response(model="gpt-4o")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).model == "gpt-4o"

    def test_finish_reason_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(finish_reason="stop")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).finish_reason == "stop"

    def test_usage_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        llm._client.post = AsyncMock(return_value=mock_resp)

        usage = asyncio.run(llm._generate(_request())).usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15


class TestRestOpenAIStream:
    def test_yields_delta_content(self):
        llm = _llm()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]}),
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
            ": keep-alive",
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hi"}}]}),
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

    def test_null_delta_content_skipped(self):
        llm = _llm()
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {}}]}),  # no content key
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
        assert chunks == ["Hello"]


class TestRestOpenAIListModels:
    def test_returns_sorted_model_ids(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]}
        llm._client.get = AsyncMock(return_value=mock_resp)

        models = asyncio.run(llm.list_models())
        assert models == sorted(["gpt-4o", "gpt-3.5-turbo"])
