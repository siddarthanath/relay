# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
import json
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Literal, overload, List, Dict

# Third Party Library
import jsonschema

# Private Library
from relay.llm.schemas import LlmMessage, LlmRequest, LlmResponse, ContentPart, TextPart
from relay.llm.errors import (
    RelayError,
    AuthError,
    RateLimitError,
    OverloadedError,
    BadRequestError,
    TimeoutError,
    ProviderError,
)

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

# Errors on which a retry may succeed. Auth/BadRequest are deterministic failures and are never retried.
RETRYABLE_ERRORS = (RateLimitError, OverloadedError, TimeoutError)

# Retry policy: a couple of attempts with exponential backoff + jitter, capped.
_MAX_ATTEMPTS = 3
_BASE_DELAY = 0.5
_MAX_DELAY = 8.0

# Name used for the schema-carrying tool on providers that surface structured
# output through forced tool-use (Anthropic).
STRUCTURED_TOOL_NAME = "structured_output"


class BaseLlm(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_provider: str, api_key: str, model_name: str | None = None) -> None:
        self.model_provider = model_provider
        self.model_name = model_name
        # This indirectly handles the context manager creation since this is sync
        self._client = self._create_client(api_key)

    @property
    def client(self):
        """Return the provider-specific API client instance."""
        return self._client

    async def __aenter__(self) -> "BaseLlm":
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> bool:
        if hasattr(self._client, "aclose"):
            await self._client.aclose()
        return False

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
            RelayError: Typed, provider-agnostic errors (rate limit, auth, etc.)

        """
        # Streaming is not part of the marking contract, so it is neither retried
        # nor error-mapped here (its per-provider adapters keep their own behaviour).
        return self._stream(request) if stream else await self._generate_with_retries(request)

    async def _generate_with_retries(self, request: LlmRequest) -> LlmResponse:
        """Run the non-streaming generate with exponential backoff + jitter on retryable errors."""
        last_error: RelayError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return await self._generate(request)
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt)) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)
        # Loop always returns or raises; this satisfies type-checkers.
        assert last_error is not None
        raise last_error

    def _base_kwargs(self, request: LlmRequest) -> dict:
        """Common generation parameters shared across all providers."""
        return {
            "model": self.model_name,
            "temperature": request.temperature,
            "max_tokens":  request.max_tokens,
        }

    # ───────────────────────────── Shared multimodal / structured helpers ───────────────────────────── #

    @staticmethod
    def _parts(message: LlmMessage) -> List[ContentPart]:
        """Normalise a message's content into a list of parts.

        A plain-string content is treated as a single text part, preserving
        backward-compatibility for text-only callers.
        """
        if isinstance(message.content, str):
            return [TextPart(text=message.content)]
        return message.content

    @staticmethod
    def _validate_structured(data: dict, schema: dict) -> dict:
        """Validate ``data`` against ``schema``; raise BadRequestError if it does not conform."""
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            raise BadRequestError(
                f"Structured output did not conform to the requested schema: {e.message}"
            ) from e
        except jsonschema.SchemaError as e:
            raise BadRequestError(f"Invalid structured_output_schema: {e.message}") from e
        return data

    @classmethod
    def _parse_structured_text(cls, text: str, schema: dict) -> dict:
        """Parse a JSON string returned by the model and validate it against the schema."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise BadRequestError("Structured output was not valid JSON") from e
        return cls._validate_structured(data, schema)

    # ───────────────────────────── Shared error mapping ───────────────────────────── #

    @staticmethod
    def _error_for_status(status: int | None, exc: Exception) -> RelayError:
        """Map an HTTP status code to the appropriate typed Relay error.

        429 → RateLimitError; 401/403 → AuthError; 529/503/overloaded → OverloadedError;
        400 → BadRequestError; anything else → ProviderError.
        """
        message = str(exc)
        if status == 429:
            return RateLimitError(message)
        if status in (401, 403):
            return AuthError(message)
        if status in (529, 503):
            return OverloadedError(message)
        if status == 400:
            return BadRequestError(message)
        return ProviderError(message)

    @abstractmethod
    def _create_client(self, api_key: str):
        """Create and return the provider-specific API client instance."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def _generate(self, request: LlmRequest) -> LlmResponse:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def _stream(self, request: LlmRequest) -> AsyncGenerator[str, None]:
        raise NotImplementedError("Subclasses must implement this method")
        yield  # makes this an async generator in the ABC, matching subclass contract

    @abstractmethod
    def _convert_messages(self, messages: list[LlmMessage]) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    async def list_models(self) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")
