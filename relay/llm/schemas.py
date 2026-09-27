# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from typing import List, Literal, Union, Annotated
from enum import Enum

# Third Party Library
from pydantic import BaseModel, Field

# Private Library

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #
class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class TextPart(BaseModel):
    """A plain-text part of a multimodal message."""

    type: Literal["text"] = "text"
    text: str = Field(description="Text content")


class ImagePart(BaseModel):
    """An image part of a multimodal message.

    ``data`` holds the raw image bytes already base64-encoded as an ASCII string
    (the same representation every provider ultimately wants on the wire).
    """

    type: Literal["image"] = "image"
    mime_type: str = Field(description="Image media type, e.g. 'image/png' or 'image/jpeg'")
    data: str = Field(description="Base64-encoded image bytes")


# Discriminated on ``type`` so pydantic can parse plain dicts into the right part.
ContentPart = Annotated[Union[TextPart, ImagePart], Field(discriminator="type")]


class LlmMessage(BaseModel):
    """Single message in conversation history.

    ``content`` accepts either a plain string (a single text part, kept for full
    backward-compatibility) or a list of :class:`ContentPart` for multimodal input.
    """

    role: Role = Field(
        description="Message role: user, assistant, or system"
    )
    content: Union[str, List[ContentPart]] = Field(
        description="Message content: plain text or a list of text/image parts"
    )

class LlmRequest(BaseModel):
    """Request to LLM provider.

    User-provided parameters for a single generation request.
    Does NOT include API keys (those come from LLMConfig).
    """

    # Required user inputs
    messages: List[LlmMessage] = Field(
        description="Conversation history including current user message"
    )

    # Generation parameters (user can override defaults)
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens to generate"
    )

    # Advanced features
    system_prompt: str | None  = None
    structured_output_schema: dict | None = Field(
        default=None, description="JSON schema for structured output generation"
    )

    # Provider-specific parameters (optional, provider-dependent)
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling (OpenAI, Google)"
    )
    top_k: int | None = Field(
        default=None, ge=1, description="Top-k sampling (Google only)"
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty (OpenAI only)"
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty (OpenAI only)"
    )

class Usage(BaseModel):
    """Typed token-usage stats.

    This is the token-tracking contract and is built identically across every
    provider adapter, so callers can rely on the same shape regardless of backend.
    """

    prompt_tokens: int = Field(description="Tokens consumed by the prompt/input")
    completion_tokens: int = Field(description="Tokens generated in the completion/output")
    total_tokens: int = Field(description="Total tokens (prompt + completion)")

class LlmResponse(BaseModel):
    """Response from LLM provider."""

    content: str = Field(description="Generated response text (raw text, even when structured output was requested)")
    model: str = Field(description="Model identifier used")
    finish_reason: str = Field(description="Why generation stopped (e.g., 'stop', 'length')")
    usage: Usage = Field(description="Typed token usage stats")
    structured: dict | None = Field(
        default=None,
        description="Parsed + schema-validated JSON when a structured_output_schema was requested; None otherwise",
    )
