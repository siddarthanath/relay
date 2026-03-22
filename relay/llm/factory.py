# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from typing import Literal, Dict

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.anthropic import NativeAnthropicLlm
from relay.llm.google import NativeGoogleLlm
from relay.llm.openai import NativeOpenAILlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

LlmProviderType = Literal["google", "openai", "anthropic"]


class LlmProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers: Dict[LlmProviderType, type[BaseLlm]] = {
        "google": NativeGoogleLlm,
        "openai": NativeOpenAILlm,
        "anthropic": NativeAnthropicLlm,
    }

    @classmethod
    def create(
        cls, provider_type: LlmProviderType, api_key: str, model_name: str | None = None
    ) -> BaseLlm:
        """Create LLM provider instance.

        Args:
            provider_type: Type of provider ("google", "openai", etc.)
            api_key: API key for the provider
            model_name: Optional model name override (uses provider default if not specified)

        Returns:
            Initialized LLM provider instance

        Raises:
            ValueError: If provider_type is not supported

        """
        if provider_type not in cls._providers:
            supported = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unsupported provider type: {provider_type}. Supported: {supported}"
            )

        provider_class = cls._providers[provider_type]

        if model_name:
            return provider_class(api_key=api_key, model_name=model_name)
        else:
            return provider_class(api_key=api_key)

    @classmethod
    def register_provider(
        cls, provider_type: str, provider_class: type[BaseLlm]
    ) -> None:
        """Register a new LLM provider type.

        Allows extending the factory with custom providers.

        Args:
            provider_type: Identifier for the provider
            provider_class: Provider class implementing BaseLlm

        """
        cls._providers[provider_type] = provider_class
