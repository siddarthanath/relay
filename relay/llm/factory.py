# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from typing import Literal, Dict, Tuple

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.native.sdk.anthropic import SdkAnthropicLlm
from relay.llm.native.sdk.google import SdkGoogleLlm
from relay.llm.native.sdk.openai import SdkOpenAILlm
from relay.llm.native.rest.anthropic import RestAnthropicLlm
from relay.llm.native.rest.google import RestGoogleLlm
from relay.llm.native.rest.openai import RestOpenAILlm

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

LlmModelProviderTypes = Literal["google", "openai", "anthropic"]
LlmImplementationTypes = Literal["sdk", "rest"]


class LlmProviderFactory:
    """Factory for creating LLM provider instances."""

    _PROVIDER_REGISTRY: Dict[Tuple[str, str], type[BaseLlm]] = {
        ("sdk", "anthropic"): SdkAnthropicLlm,
        ("sdk", "google"): SdkGoogleLlm,
        ("sdk", "openai"): SdkOpenAILlm,
        ("rest", "anthropic"): RestAnthropicLlm,
        ("rest", "google"): RestGoogleLlm,
        ("rest", "openai"): RestOpenAILlm,
    }

    @classmethod
    def create(
        cls,
        provider_type:  LlmModelProviderTypes,
        api_key: str,
        model_name: str | None = None,
        implementation: LlmImplementationTypes = "sdk",
    ) -> BaseLlm:
        """Create an LLM provider instance.

        Args:
            provider_type:  Model provider ("google", "openai", "anthropic")
            api_key: API key for the provider
            model_name: Model name override; pass None to call list_models() first
            implementation: Backend to use ("sdk" or "rest"); defaults to "sdk"

        Returns:
            Initialised LLM provider instance.

        Raises:
            ValueError: If the provider/implementation combination is not supported

        """
        key = (implementation, provider_type)
        if key not in cls._PROVIDER_REGISTRY:
            raise ValueError(
                f"Unsupported combination: provider='{provider_type}', implementation='{implementation}'"
            )

        provider_class = cls._PROVIDER_REGISTRY[key]
        return provider_class(api_key=api_key, model_name=model_name)

    @classmethod
    def register_provider(
        cls,
        provider_type: LlmModelProviderTypes,
        provider_class: type[BaseLlm],
        implementation: LlmImplementationTypes = "sdk",
    ) -> None:
        """Register a custom LLM provider."""
        cls._PROVIDER_REGISTRY[(implementation, provider_type)] = provider_class
