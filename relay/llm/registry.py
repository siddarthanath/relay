# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Private Library
from relay.llm.base import BaseLlm
from relay.llm.factory import LlmProviderFactory, LlmModelProviderTypes, LlmImplementationTypes
from relay.utils.file import load_env_file

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Maps known env var names → provider identifiers. GEMINI_API_KEY is treated as an alias for google.
_ENV_KEY_MAP: Dict[str, LlmModelProviderTypes] = {
    "ANTHROPIC_API_KEY": "anthropic",
    "OPENAI_API_KEY": "openai",
    "GOOGLE_API_KEY": "google",
    "GEMINI_API_KEY": "google",
}

_ALL_IMPLEMENTATIONS: List[LlmImplementationTypes] = ["sdk", "rest"]


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
        self._instances: Dict[Tuple[str, str], BaseLlm] = {}

        if env_file is not None:
            load_env_file(Path(env_file))

        self._build()

    def get(
        self,
        provider: LlmModelProviderTypes,
        implementation: LlmImplementationTypes = "sdk",
    ) -> BaseLlm:
        """Return a pre-instantiated LLM for the requested provider/backend.

        Args:
            provider:       One of "anthropic", "openai", "google".
            implementation: Backend to use — "sdk" (default) or "rest".

        Returns:
            Ready-to-use BaseLlm instance.

        Raises:
            KeyError: If no instance is registered for this combination (missing API key
                      or unsupported provider/implementation pair).
        """
        key = (implementation, provider)
        if key not in self._instances:
            available = [f"{impl}/{prov}" for impl, prov in self._instances] or ["none — check your API keys"]
            raise KeyError(
                f"No registered instance for provider='{provider}', implementation='{implementation}'. "
                f"Available: {available}"
            )
        return self._instances[key]

    @property
    def available(self) -> List[Tuple[str, str]]:
        """List of (implementation, provider) tuples currently registered."""
        return list(self._instances.keys())

    def _build(self) -> None:
        seen: set[str] = set()
        for env_var, provider in _ENV_KEY_MAP.items():
            if provider in seen:
                continue
            api_key = os.environ.get(env_var)
            if not api_key:
                continue
            seen.add(provider)
            for impl in _ALL_IMPLEMENTATIONS:
                try:
                    self._instances[(impl, provider)] = LlmProviderFactory.create(
                        provider_type=provider,
                        api_key=api_key,
                        model_name=self._model_names.get(provider),
                        implementation=impl,
                    )
                except ValueError:
                    pass