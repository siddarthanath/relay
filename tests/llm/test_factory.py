# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from unittest.mock import MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.factory import LlmProviderFactory
from relay.llm.base import BaseLlm
from relay.llm.providers.anthropic import AnthropicLlm
from relay.llm.providers.google import GoogleLlm
from relay.llm.providers.openai import OpenAILlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _patch(cls):
    return patch.object(cls, "_create_client", return_value=MagicMock())


class TestLlmProviderFactory:
    def test_create_anthropic(self):
        with _patch(AnthropicLlm):
            llm = LlmProviderFactory.create("anthropic", "fake-key")
        assert isinstance(llm, AnthropicLlm)

    def test_create_google(self):
        with _patch(GoogleLlm):
            llm = LlmProviderFactory.create("google", "fake-key")
        assert isinstance(llm, GoogleLlm)

    def test_create_openai(self):
        with _patch(OpenAILlm):
            llm = LlmProviderFactory.create("openai", "fake-key")
        assert isinstance(llm, OpenAILlm)

    def test_model_name_passed_through(self):
        with _patch(OpenAILlm):
            llm = LlmProviderFactory.create("openai", "fake-key", model_name="gpt-4o")
        assert llm.model_name == "gpt-4o"

    def test_model_name_none_by_default(self):
        with _patch(AnthropicLlm):
            llm = LlmProviderFactory.create("anthropic", "fake-key")
        assert llm.model_name is None

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            LlmProviderFactory.create("unknown_provider", "fake-key")

    def test_register_custom_provider(self):
        class DummyLlm(BaseLlm):
            def __init__(self, api_key, model_name=None):
                self.model_provider = "custom"
                self.model_name = model_name
                self._client = MagicMock()

            def _create_client(self, api_key): return MagicMock()
            async def _generate(self, request): ...
            async def _stream(self, request): ...
            def _convert_messages(self, messages): return []
            def list_models(self): return []

        LlmProviderFactory.register_provider("custom", DummyLlm)
        llm = LlmProviderFactory.create("custom", "fake-key")
        assert isinstance(llm, DummyLlm)

        # Cleanup: restore original registry entry
        del LlmProviderFactory._PROVIDER_REGISTRY["custom"]

    def test_register_replaces_existing(self):
        original = LlmProviderFactory._PROVIDER_REGISTRY["anthropic"]

        class ReplacementLlm(BaseLlm):
            def __init__(self, api_key, model_name=None):
                self.model_provider = "anthropic"
                self.model_name = model_name
                self._client = MagicMock()

            def _create_client(self, api_key): return MagicMock()
            async def _generate(self, request): ...
            async def _stream(self, request): ...
            def _convert_messages(self, messages): return []
            def list_models(self): return []

        LlmProviderFactory.register_provider("anthropic", ReplacementLlm)
        llm = LlmProviderFactory.create("anthropic", "fake-key")
        assert isinstance(llm, ReplacementLlm)

        # Restore original
        LlmProviderFactory._PROVIDER_REGISTRY["anthropic"] = original
