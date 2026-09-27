# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library

# Third Party Library

# Private Library

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #
# Typed error hierarchy for Relay. These classes are declared here only; adapters are
# wired to raise them in a subsequent change.


class RelayError(Exception):
    """Base class for all Relay errors."""


class AuthError(RelayError):
    """Authentication failed (invalid or missing API key)."""


class RateLimitError(RelayError):
    """Provider rate limit was exceeded."""


class OverloadedError(RelayError):
    """Provider is temporarily overloaded and could not serve the request."""


class BadRequestError(RelayError):
    """The request was malformed or invalid."""


class TimeoutError(RelayError):
    """The request timed out."""


class ProviderError(RelayError):
    """A generic, unclassified provider-side error."""
