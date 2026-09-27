# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import os
from pathlib import Path
from typing import Dict, List

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.factory import LlmProviderFactory, LlmModelProviderTypes
from relay.utils.file import load_env_file

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Maps known env var names → provider identifiers. GEMINI_API_KEY is treated as an alias for google.
_ENV_KEY_MAP: Dict[str, LlmModelProviderTypes] = {
    "ANTHROPIC_API_KEY": "anthropic",
    "OPENAI_API_KEY": "openai",
    "GOOGLE_API_KEY": "google",
    "GEMINI_API_KEY": "google",
}


class LlmProviderRegistry:
    """Pre-instantiated LLM registry populated from environment variables or a .env file,
    utilising the factory
    """

    def __init__(
        self,
        env_file: str | Path | None = None,
        model_names: Dict[LlmModelProviderTypes, str] | None = None,
    ) -> None:
        """Initialise the registry.

        Args:
            env_file: Optional path to a .env file. Variables already set in the
                         environment are NOT overwritten (os.environ.setdefault semantics).
            model_names: Optional mapping of provider → model name override, e.g.
                         {"anthropic": "claude-opus-4-6", "openai": "gpt-4o"}.
                         Providers omitted here will have model_name=None (call
                         list_models() on the instance to discover available models).

        Raises:
            FileNotFoundError: If env_file is provided but does not exist.
        """
        self._model_names: Dict[str, str] = model_names or {}
        self._instances: Dict[str, BaseLlm] = {}

        if env_file is not None:
            load_env_file(Path(env_file))

        self._build()

    def get(self, provider: LlmModelProviderTypes) -> BaseLlm:
        """Return a pre-instantiated LLM for the requested provider.

        Args:
            provider: One of "anthropic", "openai", "google".

        Returns:
            Ready-to-use BaseLlm instance.

        Raises:
            KeyError: If no instance is registered for this provider (missing API key
                      or unsupported provider).
        """
        if provider not in self._instances:
            available = list(self._instances) or ["none — check your API keys"]
            raise KeyError(
                f"No registered instance for provider='{provider}'. "
                f"Available: {available}"
            )
        return self._instances[provider]

    @property
    def available(self) -> List[str]:
        """List of provider names currently registered."""
        return list(self._instances.keys())

    def _build(self) -> None:
        for env_var, provider in _ENV_KEY_MAP.items():
            if provider in self._instances:
                continue
            api_key = os.environ.get(env_var)
            if not api_key:
                continue
            self._instances[provider] = LlmProviderFactory.create(
                provider=provider,
                api_key=api_key,
                model_name=self._model_names.get(provider),
            )
