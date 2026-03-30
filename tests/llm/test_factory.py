# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from unittest.mock import MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.factory import LlmProviderFactory
from relay.llm.base import BaseLlm
from relay.llm.native.sdk.anthropic import NativeSdkAnthropicLlm
from relay.llm.native.sdk.google import NativeSdkGoogleLlm
from relay.llm.native.sdk.openai import NativeSdkOpenAILlm
from relay.llm.native.rest.anthropic import NativeRestAnthropicLlm
from relay.llm.native.rest.google import NativeRestGoogleLlm
from relay.llm.native.rest.openai import NativeRestOpenAILlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

def _patch(cls):
    return patch.object(cls, "_create_client", return_value=MagicMock())


class TestLlmProviderFactory:
    def test_create_sdk_anthropic(self):
        with _patch(NativeSdkAnthropicLlm):
            llm = LlmProviderFactory.create("anthropic", "fake-key", implementation="sdk")
        assert isinstance(llm, NativeSdkAnthropicLlm)

    def test_create_sdk_google(self):
        with _patch(NativeSdkGoogleLlm):
            llm = LlmProviderFactory.create("google", "fake-key", implementation="sdk")
        assert isinstance(llm, NativeSdkGoogleLlm)

    def test_create_sdk_openai(self):
        with _patch(NativeSdkOpenAILlm):
            llm = LlmProviderFactory.create("openai", "fake-key", implementation="sdk")
        assert isinstance(llm, NativeSdkOpenAILlm)

    def test_create_rest_anthropic(self):
        with _patch(NativeRestAnthropicLlm):
            llm = LlmProviderFactory.create("anthropic", "fake-key", implementation="rest")
        assert isinstance(llm, NativeRestAnthropicLlm)

    def test_create_rest_google(self):
        with _patch(NativeRestGoogleLlm):
            llm = LlmProviderFactory.create("google", "fake-key", implementation="rest")
        assert isinstance(llm, NativeRestGoogleLlm)

    def test_create_rest_openai(self):
        with _patch(NativeRestOpenAILlm):
            llm = LlmProviderFactory.create("openai", "fake-key", implementation="rest")
        assert isinstance(llm, NativeRestOpenAILlm)

    def test_default_implementation_is_sdk(self):
        with _patch(NativeSdkAnthropicLlm):
            llm = LlmProviderFactory.create("anthropic", "fake-key")
        assert isinstance(llm, NativeSdkAnthropicLlm)

    def test_model_name_passed_through(self):
        with _patch(NativeSdkOpenAILlm):
            llm = LlmProviderFactory.create("openai", "fake-key", model_name="gpt-4o")
        assert llm.model_name == "gpt-4o"

    def test_model_name_none_by_default(self):
        with _patch(NativeSdkAnthropicLlm):
            llm = LlmProviderFactory.create("anthropic", "fake-key")
        assert llm.model_name is None

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported combination"):
            LlmProviderFactory.create("unknown_provider", "fake-key")

    def test_invalid_implementation_raises(self):
        with pytest.raises(ValueError, match="Unsupported combination"):
            LlmProviderFactory.create("anthropic", "fake-key", implementation="grpc")

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

        LlmProviderFactory.register_provider("anthropic", DummyLlm, implementation="grpc")
        llm = LlmProviderFactory.create("anthropic", "fake-key", implementation="grpc")
        assert isinstance(llm, DummyLlm)

        # Cleanup: restore original registry entry
        del LlmProviderFactory._PROVIDER_REGISTRY[("grpc", "anthropic")]

    def test_register_replaces_existing(self):
        original = LlmProviderFactory._PROVIDER_REGISTRY[("sdk", "anthropic")]

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

        LlmProviderFactory.register_provider("anthropic", ReplacementLlm, implementation="sdk")
        llm = LlmProviderFactory.create("anthropic", "fake-key", implementation="sdk")
        assert isinstance(llm, ReplacementLlm)

        # Restore original
        LlmProviderFactory._PROVIDER_REGISTRY[("sdk", "anthropic")] = original
