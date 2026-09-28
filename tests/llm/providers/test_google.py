# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest
import httpx
from google.genai import errors as genai_errors

# Private Library
from relay.llm.providers.google import GoogleLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role, ImagePart
from relay.llm.errors import RateLimitError, AuthError, OverloadedError, BadRequestError, ProviderError, TimeoutError

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="gemini-2.0-flash") -> GoogleLlm:
    with patch.object(GoogleLlm, "_create_client", return_value=MagicMock()):
        return GoogleLlm(api_key="fake-key", model_name=model_name)


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


class TestGoogleLlmInit:
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
        assert response.usage.prompt_tokens == 8
        assert response.usage.completion_tokens == 4
        assert response.usage.total_tokens == 12

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
        assert response.usage.prompt_tokens == 0
        assert response.usage.completion_tokens == 0
        assert response.usage.total_tokens == 0


SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}},
    "required": ["score"],
}


class TestSdkGoogleStructuredOutput:
    def test_config_sets_json_mode_and_schema(self):
        llm = _llm()
        config = llm._build_config(_request(structured_output_schema=SCHEMA))
        assert config.response_mime_type == "application/json"
        assert config.response_schema is not None

    def test_config_no_json_mode_without_schema(self):
        llm = _llm()
        config = llm._build_config(_request())
        assert config.response_mime_type is None

    def test_structured_populated_and_validated(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(
            return_value=_mock_generate_response(text='{"score": 7}')
        )
        response = asyncio.run(llm._generate(_request(structured_output_schema=SCHEMA)))
        assert response.structured == {"score": 7}

    def test_structured_none_without_schema(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(return_value=_mock_generate_response())
        response = asyncio.run(llm._generate(_request()))
        assert response.structured is None

    def test_malformed_json_raises_bad_request(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(
            return_value=_mock_generate_response(text="not json")
        )
        with pytest.raises(BadRequestError):
            asyncio.run(llm._generate(_request(structured_output_schema=SCHEMA)))

    def test_non_conforming_raises_bad_request(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(
            return_value=_mock_generate_response(text='{"score": "high"}')
        )
        with pytest.raises(BadRequestError):
            asyncio.run(llm._generate(_request(structured_output_schema=SCHEMA)))


class TestSdkGoogleImageConversion:
    def test_image_part_native_inline_data(self):
        llm = _llm()
        msg = LlmMessage(role=Role.user, content=[ImagePart(mime_type="image/png", data="YWJj")])
        result = llm._convert_messages([msg])
        assert result == [{
            "role": "user",
            "parts": [{"inline_data": {"mime_type": "image/png", "data": "YWJj"}}],
        }]

    def test_str_content_unchanged(self):
        llm = _llm()
        result = llm._convert_messages([LlmMessage(role=Role.user, content="Hello")])
        assert result == [{"role": "user", "parts": [{"text": "Hello"}]}]


class TestSdkGoogleErrorMapping:
    def test_429_maps_to_rate_limit(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(side_effect=genai_errors.APIError(429, {}, None))
        with pytest.raises(RateLimitError):
            asyncio.run(llm._generate(_request()))

    def test_401_maps_to_auth(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(side_effect=genai_errors.APIError(401, {}, None))
        with pytest.raises(AuthError):
            asyncio.run(llm._generate(_request()))

    def test_503_maps_to_overloaded(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(side_effect=genai_errors.APIError(503, {}, None))
        with pytest.raises(OverloadedError):
            asyncio.run(llm._generate(_request()))

    def test_500_maps_to_provider_error(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(side_effect=genai_errors.APIError(500, {}, None))
        with pytest.raises(ProviderError):
            asyncio.run(llm._generate(_request()))

    def test_timeout_maps_to_timeout(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(side_effect=httpx.TimeoutException("slow"))
        with pytest.raises(TimeoutError):
            asyncio.run(llm._generate(_request()))


class TestSdkGoogleRetry:
    def test_retries_then_succeeds(self):
        llm = _llm()
        llm._client.aio.models.generate_content = AsyncMock(
            side_effect=[genai_errors.APIError(503, {}, None), _mock_generate_response(text="ok")]
        )
        with patch("relay.llm.base.asyncio.sleep", new=AsyncMock()):
            response = asyncio.run(llm.generate(_request()))
        assert response.content == "ok"
        assert llm._client.aio.models.generate_content.call_count == 2


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


# Regression: usage parsing must survive thinking models (None counts) and split thinking out.
from types import SimpleNamespace
from relay.llm.providers.google import GoogleLlm as _G


class TestUsageParsing:
    def test_thinking_model_none_candidates_is_coalesced_and_split(self):
        # A thinking model that emitted no visible text: candidates=None, thoughts=50, total=None.
        um = SimpleNamespace(
            prompt_token_count=10, candidates_token_count=None,
            thoughts_token_count=50, total_token_count=None,
        )
        u = _G._usage(um)
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 0        # visible output only
        assert u.thinking_tokens == 50         # split out, not folded into completion
        assert u.total_tokens == 60            # prompt + completion + thinking (derived)

    def test_non_thinking_model_has_zero_thinking(self):
        um = SimpleNamespace(
            prompt_token_count=8, candidates_token_count=4, total_token_count=12,
        )  # no thoughts_token_count attribute at all
        u = _G._usage(um)
        assert (u.prompt_tokens, u.completion_tokens, u.thinking_tokens, u.total_tokens) == (8, 4, 0, 12)

    def test_missing_usage_metadata_is_all_zero(self):
        u = _G._usage(None)
        assert (u.prompt_tokens, u.completion_tokens, u.thinking_tokens, u.total_tokens) == (0, 0, 0, 0)
