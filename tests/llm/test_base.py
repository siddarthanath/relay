# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.native.sdk.anthropic import SdkAnthropicLlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _llm() -> SdkAnthropicLlm:
    with patch.object(SdkAnthropicLlm, "_create_client", return_value=MagicMock()):
        return SdkAnthropicLlm(api_key="fake-key", model_name="claude-opus-4-6")


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
