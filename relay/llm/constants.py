# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library

# Third Party Library

# Private Library

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

_PROVIDERS       = ["native"]
_IMPLEMENTATIONS = ["sdk", "rest"]
_MODELS          = ["anthropic", "google", "openai"]

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "google":    "gemini-2.0-flash",
    "openai":    "gpt-4o",
}
