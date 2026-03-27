# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

# Third Party Library
import pytest

# Private Library
from relay.llm.registry import LlmProviderRegistry
from relay.llm.native.sdk.anthropic import SdkAnthropicLlm
from relay.llm.native.sdk.google import SdkGoogleLlm
from relay.llm.native.sdk.openai import SdkOpenAILlm
from relay.llm.native.rest.anthropic import RestAnthropicLlm
from relay.llm.native.rest.google import RestGoogleLlm
from relay.llm.native.rest.openai import RestOpenAILlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Helpers

_ALL_CLASSES = [
    SdkAnthropicLlm,
    SdkGoogleLlm,
    SdkOpenAILlm,
    RestAnthropicLlm,
    RestGoogleLlm,
    RestOpenAILlm,
]


def _patch_all_clients():
    return [patch.object(cls, "_create_client", return_value=MagicMock()) for cls in _ALL_CLASSES]


def _registry_with_all_keys(monkeypatch, **kwargs) -> LlmProviderRegistry:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fake")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-fake")
    with ExitStack() as stack:
        for p in _patch_all_clients():
            stack.enter_context(p)
        return LlmProviderRegistry(**kwargs)


class TestLlmProviderRegistry:
    def test_empty_when_no_env_keys(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        registry = LlmProviderRegistry()
        assert registry.available == []

    def test_get_on_empty_registry_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        registry = LlmProviderRegistry()
        with pytest.raises(KeyError):
            registry.get("anthropic")

    def test_registers_sdk_and_rest_for_each_key(self, monkeypatch):
        registry = _registry_with_all_keys(monkeypatch)
        assert ("sdk", "anthropic") in registry.available
        assert ("rest", "anthropic") in registry.available
        assert ("sdk", "openai") in registry.available
        assert ("rest", "openai") in registry.available
        assert ("sdk", "google") in registry.available
        assert ("rest", "google") in registry.available

    def test_get_returns_sdk_by_default(self, monkeypatch):
        registry = _registry_with_all_keys(monkeypatch)
        llm = registry.get("anthropic")
        assert isinstance(llm, SdkAnthropicLlm)

    def test_get_rest_implementation(self, monkeypatch):
        registry = _registry_with_all_keys(monkeypatch)
        llm = registry.get("anthropic", implementation="rest")
        assert isinstance(llm, RestAnthropicLlm)

    def test_get_google_sdk(self, monkeypatch):
        registry = _registry_with_all_keys(monkeypatch)
        llm = registry.get("google")
        assert isinstance(llm, SdkGoogleLlm)

    def test_get_openai_rest(self, monkeypatch):
        registry = _registry_with_all_keys(monkeypatch)
        llm = registry.get("openai", implementation="rest")
        assert isinstance(llm, RestOpenAILlm)

    def test_get_missing_provider_raises_key_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with ExitStack() as stack:
            for p in _patch_all_clients():
                stack.enter_context(p)
            registry = LlmProviderRegistry()
        with pytest.raises(KeyError):
            registry.get("openai")

    def test_gemini_key_registers_as_google(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake")
        with ExitStack() as stack:
            for p in _patch_all_clients():
                stack.enter_context(p)
            registry = LlmProviderRegistry()
        assert ("sdk", "google") in registry.available

    def test_gemini_key_does_not_register_twice(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "goog-fake")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake")
        with ExitStack() as stack:
            for p in _patch_all_clients():
                stack.enter_context(p)
            registry = LlmProviderRegistry()
        google_entries = [k for k in registry.available if k[1] == "google"]
        assert len(google_entries) == 2  # exactly sdk + rest, not 4

    def test_model_names_passed_through(self, monkeypatch):
        registry = _registry_with_all_keys(
            monkeypatch,
            model_names={"anthropic": "claude-opus-4-6", "openai": "gpt-4o"},
        )
        assert registry.get("anthropic").model_name == "claude-opus-4-6"
        assert registry.get("openai").model_name == "gpt-4o"

    def test_env_file_loaded(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-from-file\n")
        with ExitStack() as stack:
            for p in _patch_all_clients():
                stack.enter_context(p)
            registry = LlmProviderRegistry(env_file=env_file)
        assert ("sdk", "anthropic") in registry.available

    def test_env_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LlmProviderRegistry(env_file=tmp_path / "missing.env")

    def test_env_file_does_not_overwrite_existing_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-shell")
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-from-file\n")
        with ExitStack() as stack:
            for p in _patch_all_clients():
                stack.enter_context(p)
            registry = LlmProviderRegistry(env_file=env_file)
        # Instance should exist (from shell key), and model_provider is set correctly
        assert ("sdk", "anthropic") in registry.available

    def test_available_returns_list_of_tuples(self, monkeypatch):
        registry = _registry_with_all_keys(monkeypatch)
        for entry in registry.available:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            impl, provider = entry
            assert impl in ("sdk", "rest")
            assert provider in ("anthropic", "openai", "google")
