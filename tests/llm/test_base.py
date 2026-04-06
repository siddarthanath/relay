# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.sdk.anthropic import NativeSdkAnthropicLlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _llm() -> NativeSdkAnthropicLlm:
    with patch.object(NativeSdkAnthropicLlm, "_create_client", return_value=MagicMock()):
        return NativeSdkAnthropicLlm(api_key="fake-key", model_name="claude-opus-4-6")


class TestAsyncContextManager:
    def test_aenter_returns_self(self):
        llm = _llm()
        result = asyncio.run(llm.__aenter__())
        assert result is llm

    def test_aexit_calls_aclose(self):
        llm = _llm()
        llm._client.aclose = AsyncMock()
        asyncio.run(llm.__aexit__(None, None, None))
        llm._client.aclose.assert_called_once()

    def test_aexit_returns_false(self):
        llm = _llm()
        llm._client.aclose = AsyncMock()
        result = asyncio.run(llm.__aexit__(None, None, None))
        assert result is False

    def test_aexit_safe_when_no_aclose(self):
        llm = _llm()
        llm._client = object()  # plain object with no aclose
        asyncio.run(llm.__aexit__(None, None, None))  # should not raise

    def test_context_manager_closes_client(self):
        async def _run():
            with patch.object(NativeSdkAnthropicLlm, "_create_client", return_value=MagicMock()):
                llm = NativeSdkAnthropicLlm(api_key="fake-key", model_name="claude-opus-4-6")
                llm._client.aclose = AsyncMock()
                async with llm as ctx:
                    assert ctx is llm
                llm._client.aclose.assert_called_once()

        asyncio.run(_run())


class TestValidateModel:
    def test_returns_model_name_when_valid(self):
        llm = _llm()
        llm.list_models = AsyncMock(return_value=["claude-opus-4-6", "claude-sonnet-4-6"])
        result = asyncio.run(llm._validate_model("claude-opus-4-6"))
        assert result == "claude-opus-4-6"

    def test_raises_value_error_for_unknown_model(self):
        llm = _llm()
        llm.list_models = AsyncMock(return_value=["claude-opus-4-6"])
        with pytest.raises(ValueError, match="not available"):
            asyncio.run(llm._validate_model("gpt-4o"))

    def test_error_message_includes_model_name(self):
        llm = _llm()
        llm.list_models = AsyncMock(return_value=["claude-opus-4-6"])
        with pytest.raises(ValueError, match="unknown-model"):
            asyncio.run(llm._validate_model("unknown-model"))

    def test_error_message_includes_provider(self):
        llm = _llm()
        llm.list_models = AsyncMock(return_value=[])
        with pytest.raises(ValueError, match="anthropic"):
            asyncio.run(llm._validate_model("any-model"))

    def test_empty_model_list_always_raises(self):
        llm = _llm()
        llm.list_models = AsyncMock(return_value=[])
        with pytest.raises(ValueError):
            asyncio.run(llm._validate_model("claude-opus-4-6"))
