# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from typing import Literal, Dict

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.providers.anthropic import AnthropicLlm
from relay.llm.providers.google import GoogleLlm
from relay.llm.providers.openai import OpenAILlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

LlmModelProviderTypes = Literal["google", "openai", "anthropic"]


class LlmProviderFactory:
    """Factory for creating LLM provider instances."""

    _PROVIDER_REGISTRY: Dict[str, type[BaseLlm]] = {
        "anthropic": AnthropicLlm,
        "google": GoogleLlm,
        "openai": OpenAILlm,
    }

    @classmethod
    def create(
        cls,
        provider: LlmModelProviderTypes,
        api_key: str,
        model_name: str | None = None,
    ) -> BaseLlm:
        """Create an LLM provider instance.

        Args:
            provider:   Model provider ("google", "openai", "anthropic")
            api_key:    API key for the provider
            model_name: Model name override; pass None to call list_models() first

        Returns:
            Initialised LLM provider instance.

        Raises:
            ValueError: If the provider is not supported

        """
        if provider not in cls._PROVIDER_REGISTRY:
            raise ValueError(f"Unsupported provider: '{provider}'")

        provider_class = cls._PROVIDER_REGISTRY[provider]
        return provider_class(api_key=api_key, model_name=model_name)

    @classmethod
    def register_provider(
        cls,
        provider: LlmModelProviderTypes,
        provider_class: type[BaseLlm],
    ) -> None:
        """Register a custom LLM provider."""
        cls._PROVIDER_REGISTRY[provider] = provider_class
