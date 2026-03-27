# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.sdk.openai import SdkOpenAILlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="gpt-4o") -> SdkOpenAILlm:
    with patch.object(SdkOpenAILlm, "_create_client", return_value=MagicMock()):
        return SdkOpenAILlm(api_key="fake-key", model_name=model_name)


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[LlmMessage(role=Role.user, content="Hello")])
    return LlmRequest(**(defaults | kwargs))


def _mock_generate_response(content="Hi there", model="gpt-4o", finish_reason="stop"):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.model = model
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


class TestSdkOpenAILlmInit:
    def test_model_provider_set(self):
        assert _llm().model_provider == "openai"

    def test_model_name_set(self):
        assert _llm("gpt-4o").model_name == "gpt-4o"

    def test_model_name_none(self):
        assert _llm(model_name=None).model_name is None


class TestSdkOpenAIConvertMessages:
    def test_user_message_converted(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.user, content="Hello")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_converted(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.assistant, content="Hi")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "assistant", "content": "Hi"}]

    def test_system_message_included(self):
        # OpenAI supports system role natively — not excluded
        llm = _llm()
        msgs = [LlmMessage(role=Role.system, content="Be helpful.")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "system", "content": "Be helpful."}]

    def test_empty_messages(self):
        assert _llm()._convert_messages([]) == []

    def test_order_preserved(self):
        llm = _llm()
        msgs = [
            LlmMessage(role=Role.user, content="First"),
            LlmMessage(role=Role.assistant, content="Second"),
        ]
        result = llm._convert_messages(msgs)
        assert result[0]["content"] == "First"
        assert result[1]["content"] == "Second"


class TestSdkOpenAIBuildMessages:
    def test_no_system_prompt(self):
        llm = _llm()
        msgs = llm._build_messages(_request())
        assert msgs[0]["role"] == "user"

    def test_system_prompt_prepended(self):
        llm = _llm()
        msgs = llm._build_messages(_request(system_prompt="You are helpful."))
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1]["role"] == "user"

    def test_system_prompt_absent_when_none(self):
        llm = _llm()
        msgs = llm._build_messages(_request(system_prompt=None))
        assert not any(m["role"] == "system" for m in msgs)


class TestSdkOpenAIBuildKwargs:
    def test_top_p_included_when_set(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request(top_p=0.9))
        assert kwargs["top_p"] == 0.9

    def test_top_p_absent_when_not_set(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request())
        assert "top_p" not in kwargs

    def test_frequency_penalty_included_when_set(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request(frequency_penalty=0.5))
        assert kwargs["frequency_penalty"] == 0.5

    def test_presence_penalty_included_when_set(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request(presence_penalty=0.3))
        assert kwargs["presence_penalty"] == 0.3

    def test_stream_flag_default_false(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request())
        assert kwargs["stream"] is False

    def test_stream_flag_set_to_true(self):
        llm = _llm()
        kwargs = llm._build_kwargs(_request(), stream=True)
        assert kwargs["stream"] is True


class TestSdkOpenAIGenerate:
    def test_returns_llm_response(self):
        llm = _llm()
        llm._client.chat.completions.create = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert isinstance(response, LlmResponse)

    def test_content_mapped(self):
        llm = _llm()
        llm._client.chat.completions.create = AsyncMock(return_value=_mock_generate_response(content="Answer"))
        response = asyncio.run(llm._generate(_request()))
        assert response.content == "Answer"

    def test_model_mapped(self):
        llm = _llm()
        llm._client.chat.completions.create = AsyncMock(return_value=_mock_generate_response(model="gpt-4o"))
        response = asyncio.run(llm._generate(_request()))
        assert response.model == "gpt-4o"

    def test_finish_reason_mapped(self):
        llm = _llm()
        llm._client.chat.completions.create = AsyncMock(return_value=_mock_generate_response(finish_reason="stop"))
        response = asyncio.run(llm._generate(_request()))
        assert response.finish_reason == "stop"

    def test_usage_mapped(self):
        llm = _llm()
        llm._client.chat.completions.create = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 5
        assert response.usage["total_tokens"] == 15


class TestSdkOpenAIStream:
    def test_yields_delta_content(self):
        llm = _llm()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        llm._client.chat.completions.create = AsyncMock(return_value=_aiter(chunk1, chunk2))

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hello", " world"]

    def test_none_delta_content_skipped(self):
        llm = _llm()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        llm._client.chat.completions.create = AsyncMock(return_value=_aiter(chunk1, chunk2))

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hello"]


class TestSdkOpenAIListModels:
    def test_returns_sorted_model_ids(self):
        llm = _llm()
        m1, m2 = MagicMock(id="gpt-4o"), MagicMock(id="gpt-3.5-turbo")
        mock_result = MagicMock()
        mock_result.data = [m1, m2]
        llm._client.models.list = AsyncMock(return_value=mock_result)

        models = asyncio.run(llm.list_models())
        assert models == sorted(["gpt-4o", "gpt-3.5-turbo"])
