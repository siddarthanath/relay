# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
from typing import Dict, List
from enum import Enum

# Third Party Library
from pydantic import BaseModel, Field

# Private Library

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #
class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class LlmMessage(BaseModel):
    """Single message in conversation history."""

    role: Role = Field(
        description="Message role: user, assistant, model, or system"
    )
    content: str = Field(description="Message content text")

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
    thinking_mode: bool = Field(
        default=False, description="Enable extended thinking/reasoning (o1, Claude extended thinking)"
    )
    structured_output_schema: dict | None = Field(
        default=None, description="JSON schema for structured output generation"
    )
    tools: list[dict] | None = Field(
        default=None, description="Tool definitions for function calling (Phase 3)"
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

class LlmResponse(BaseModel):
    """Response from LLM provider."""

    content: str = Field(description="Generated response text")
    model: str = Field(description="Model identifier used")
    finish_reason: str = Field(description="Why generation stopped (e.g., 'stop', 'length')")
    usage: Dict[str, int] = Field(
        description="Token usage stats (prompt_tokens, completion_tokens, total_tokens)"
    )
