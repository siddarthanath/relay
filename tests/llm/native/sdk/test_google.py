# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.sdk.google import SdkGoogleLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="gemini-2.0-flash") -> SdkGoogleLlm:
    with patch.object(SdkGoogleLlm, "_create_client", return_value=MagicMock()):
        return SdkGoogleLlm(api_key="fake-key", model_name=model_name)


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[LlmMessage(role=Role.user, content="Hello")])
    return LlmRequest(**(defaults | kwargs))


def _mock_generate_response(text="Hi there", finish_reason_name="STOP"):
    resp = MagicMock()
    resp.text = text
    resp.candidates = [MagicMock()]
    resp.candidates[0].finish_reason.name = finish_reason_name
    resp.usage_metadata.prompt_token_count = 8
    resp.usage_metadata.candidates_token_count = 4
    resp.usage_metadata.total_token_count = 12
    return resp


class TestSdkGoogleLlmInit:
    def test_model_provider_set(self):
        assert _llm().model_provider == "google"

    def test_model_name_set(self):
        assert _llm("gemini-2.0-flash").model_name == "gemini-2.0-flash"

    def test_model_name_none(self):
        assert _llm(model_name=None).model_name is None


class TestSdkGoogleConvertMessages:
    def test_user_message_converted(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.user, content="Hello")]
        result = llm._convert_messages(msgs)
        assert result == [{"role": "user", "parts": [{"text": "Hello"}]}]

    def test_assistant_mapped_to_model(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.assistant, content="Hi")]
        result = llm._convert_messages(msgs)
        assert result[0]["role"] == "model"

    def test_system_message_excluded(self):
        llm = _llm()
        msgs = [
            LlmMessage(role=Role.system, content="Be helpful."),
            LlmMessage(role=Role.user, content="Hello"),
        ]
        result = llm._convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_content_wrapped_in_parts(self):
        llm = _llm()
        msgs = [LlmMessage(role=Role.user, content="Test")]
        result = llm._convert_messages(msgs)
        assert result[0]["parts"] == [{"text": "Test"}]

    def test_empty_messages(self):
        assert _llm()._convert_messages([]) == []


class TestSdkGoogleBuildConfig:
    def test_temperature_in_config(self):
        llm = _llm()
        config = llm._build_config(_request(temperature=0.5))
        assert config.temperature == 0.5

    def test_top_p_in_config(self):
        llm = _llm()
        config = llm._build_config(_request(top_p=0.9))
        assert config.top_p == 0.9

    def test_top_k_in_config(self):
        llm = _llm()
        config = llm._build_config(_request(top_k=40))
        assert config.top_k == 40

    def test_system_prompt_included_when_set(self):
        llm = _llm()
        config = llm._build_config(_request(system_prompt="Be concise."))
        assert config.system_instruction == "Be concise."

    def test_system_prompt_absent_when_not_set(self):
        llm = _llm()
        config = llm._build_config(_request())
        assert config.system_instruction is None


class TestSdkGoogleGenerate:
    def test_returns_llm_response(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert isinstance(response, LlmResponse)

    def test_content_mapped(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(return_value=_mock_generate_response(text="Answer"))
        response = asyncio.run(llm._generate(_request()))
        assert response.content == "Answer"

    def test_finish_reason_mapped(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(return_value=_mock_generate_response(finish_reason_name="STOP"))
        response = asyncio.run(llm._generate(_request()))
        assert response.finish_reason == "STOP"

    def test_model_is_model_name(self):
        llm = _llm("gemini-2.0-flash")
        llm._client.aio.models.generate_content = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert response.model == "gemini-2.0-flash"

    def test_usage_mapped(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert response.usage["prompt_tokens"] == 8
        assert response.usage["completion_tokens"] == 4
        assert response.usage["total_tokens"] == 12

    def test_no_candidates_finish_reason_unknown(self):
        llm = _llm()
        resp = _mock_generate_response()
        resp.candidates = []
        llm._client.aio.models.generate_content = AsyncMock(return_value=resp)
        response = asyncio.run(llm._generate(_request()))
        assert response.finish_reason == "unknown"

    def test_no_usage_metadata_defaults_to_zero(self):
        llm = _llm()
        resp = _mock_generate_response()
        resp.usage_metadata = None
        llm._client.aio.models.generate_content = AsyncMock(return_value=resp)
        response = asyncio.run(llm._generate(_request()))
        assert response.usage["prompt_tokens"] == 0
        assert response.usage["completion_tokens"] == 0
        assert response.usage["total_tokens"] == 0


class TestSdkGoogleStream:
    def test_yields_text_chunks(self):
        llm = _llm()
        chunk1 = MagicMock(text="Hello")
        chunk2 = MagicMock(text=" world")
        llm._client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter(chunk1, chunk2))

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hello", " world"]

    def test_chunks_with_empty_text_skipped(self):
        llm = _llm()
        chunk1 = MagicMock(text="Hello")
        chunk2 = MagicMock(text=None)
        chunk3 = MagicMock(text=" world")
        llm._client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter(chunk1, chunk2, chunk3))

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Hello", " world"]


class TestSdkGoogleListModels:
    def test_returns_sorted_model_names(self):
        llm = _llm()
        m1, m2 = MagicMock(), MagicMock()
        m1.name = "models/gemini-2.0-flash"
        m2.name = "models/gemini-1.5-pro"
        llm._client.aio.models.list = AsyncMock(return_value=[m1, m2])

        models = asyncio.run(llm.list_models())
        assert models == sorted(["gemini-2.0-flash", "gemini-1.5-pro"])

    def test_strips_models_prefix(self):
        llm = _llm()
        m = MagicMock()
        m.name = "models/gemini-2.0-flash"
        llm._client.aio.models.list = AsyncMock(return_value=[m])

        models = asyncio.run(llm.list_models())
        assert models == ["gemini-2.0-flash"]
