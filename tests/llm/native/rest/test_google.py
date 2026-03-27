# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.rest.google import RestGoogleLlm
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, Role

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

async def _aiter(*items):
    for item in items:
        yield item


async def _collect(gen):
    return [item async for item in gen]


def _llm(model_name="gemini-2.0-flash") -> RestGoogleLlm:
    with patch.object(RestGoogleLlm, "_create_client", return_value=MagicMock()):
        return RestGoogleLlm(api_key="fake-key", model_name=model_name)


def _request(**kwargs) -> LlmRequest:
    defaults = dict(messages=[LlmMessage(role=Role.user, content="Hello")])
    return LlmRequest(**(defaults | kwargs))


def _api_response(text="Hi there", finish_reason="STOP",
                  prompt_tokens=8, candidates_tokens=4, total_tokens=12) -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "totalTokenCount": total_tokens,
        },
    }


class TestRestGoogleLlmInit:
    def test_model_provider_set(self):
        assert _llm().model_provider == "google"

    def test_base_url(self):
        assert RestGoogleLlm.GOOGLE_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/models"


class TestRestGoogleConvertMessages:
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
            LlmMessage(role=Role.system, content="You are helpful."),
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


class TestRestGoogleBuildBody:
    def test_contents_key_present(self):
        llm = _llm()
        body = llm._build_body(_request())
        assert "contents" in body

    def test_generation_config_present(self):
        llm = _llm()
        body = llm._build_body(_request())
        assert "generationConfig" in body

    def test_temperature_in_generation_config(self):
        llm = _llm()
        body = llm._build_body(_request(temperature=0.3))
        assert body["generationConfig"]["temperature"] == 0.3

    def test_top_p_in_generation_config(self):
        llm = _llm()
        body = llm._build_body(_request(top_p=0.9))
        assert body["generationConfig"]["topP"] == 0.9

    def test_top_k_in_generation_config(self):
        llm = _llm()
        body = llm._build_body(_request(top_k=40))
        assert body["generationConfig"]["topK"] == 40

    def test_system_prompt_included_when_set(self):
        llm = _llm()
        body = llm._build_body(_request(system_prompt="Be helpful."))
        assert body["systemInstruction"] == {"parts": [{"text": "Be helpful."}]}

    def test_system_prompt_absent_when_not_set(self):
        llm = _llm()
        body = llm._build_body(_request())
        assert "systemInstruction" not in body


class TestRestGoogleGenerate:
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
        mock_resp.json.return_value = _api_response(text="Answer")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).content == "Answer"

    def test_finish_reason_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(finish_reason="STOP")
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).finish_reason == "STOP"

    def test_model_is_model_name(self):
        llm = _llm("gemini-2.0-flash")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response()
        llm._client.post = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm._generate(_request())).model == "gemini-2.0-flash"

    def test_usage_mapped(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _api_response(prompt_tokens=8, candidates_tokens=4, total_tokens=12)
        llm._client.post = AsyncMock(return_value=mock_resp)

        usage = asyncio.run(llm._generate(_request())).usage
        assert usage["prompt_tokens"] == 8
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 12

    def test_missing_usage_metadata_defaults_to_zero(self):
        llm = _llm()
        data = _api_response()
        del data["usageMetadata"]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = data
        llm._client.post = AsyncMock(return_value=mock_resp)

        usage = asyncio.run(llm._generate(_request())).usage
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0


class TestRestGoogleListModels:
    def test_returns_sorted_model_names(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "models/gemini-2.0-flash"}, {"name": "models/gemini-1.5-pro"}]}
        llm._client.get = AsyncMock(return_value=mock_resp)

        models = asyncio.run(llm.list_models())
        assert models == sorted(["gemini-2.0-flash", "gemini-1.5-pro"])

    def test_strips_models_prefix(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "models/gemini-2.0-flash"}]}
        llm._client.get = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm.list_models()) == ["gemini-2.0-flash"]

    def test_empty_models_returns_empty_list(self):
        llm = _llm()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}  # No "models" key — defaults to []
        llm._client.get = AsyncMock(return_value=mock_resp)

        assert asyncio.run(llm.list_models()) == []


class TestRestGoogleStream:
    def test_yields_text_chunks(self):
        llm = _llm()
        lines = [
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}),
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": " world"}]}}]}),
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
            "event: start",
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}),
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

    def test_malformed_chunk_skipped(self):
        llm = _llm()
        lines = [
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Good"}]}}]}),
            "data: " + json.dumps({}),  # Missing candidates — KeyError handled
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = lambda: _aiter(*lines)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        llm._client.stream.return_value = mock_cm

        chunks = asyncio.run(_collect(llm._stream(_request())))
        assert chunks == ["Good"]
