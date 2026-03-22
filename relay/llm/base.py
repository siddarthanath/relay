# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal, overload, List, Dict

# Private Library
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #
""" 
TODO:
1. Add structured output support (e.g. JSON schemas) with provider-specific parsing logic.
2. Add prompt engineering utilities (e.g. system prompt handling, few-shot examples) with provider-specific formatting.
3. Add tool/function calling support with provider-specific implementations.
"""

class BaseLlm(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_provider: str, api_key: str, model_name: str | None = None) -> None:
        self.model_provider = model_provider
        self.model_name = model_name
        self._client = self._create_client(api_key)

    @property
    def client(self):
        """Return the provider-specific API client instance."""
        return self._client

    @overload
    async def generate(self, request: LlmRequest, stream: Literal[False] = ...) -> LlmResponse: ...
    @overload
    async def generate(self, request: LlmRequest, stream: Literal[True]) -> AsyncIterator[str]: ...

    async def generate(self, request: LlmRequest, stream: bool = False) -> LlmResponse | AsyncIterator[str]:
        """Generate response from LLM.

        Args:
            request: LLM request with messages and generation parameters
            stream: Whether to stream response chunks (default: False)

        Returns:
            If stream=False: LlmResponse with complete generated content
            If stream=True: AsyncIterator yielding content chunks as strings

        Raises:
            Exception: Provider-specific errors (rate limits, auth, etc.)

        """
        return self._stream(request) if stream else await self._generate(request)
    
    def _base_kwargs(self, request: LlmRequest) -> dict:
        """Common generation parameters shared across all providers."""
        return {
            "model": self.model_name,
            "temperature": request.temperature,
            "max_tokens":  request.max_tokens,
        }
    
    def _validate_model(self, model_name: str) -> bool:
        """Validate if the specified model is available for this provider."""
        if model_name not in self.list_models():
            raise ValueError(f"Model '{model_name}' not available for provider '{self.model_provider}'")
        return model_name

    @abstractmethod
    def _create_client(self, api_key: str):
        """Create and return the provider-specific API client instance."""
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    async def _generate(self, request: LlmRequest) -> LlmResponse: 
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def _stream(self, request: LlmRequest) -> AsyncIterator[str]: 
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _convert_messages(self, messages: list[LlmMessage]) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def list_models(self) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")