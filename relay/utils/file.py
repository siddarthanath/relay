# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import os
from pathlib import Path

# Private Library

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

def load_env_file(path: str | Path) -> None:
    """This function parses a .env file and inject variables into os.environ (setdefault semantics).

    Args:
        path (str | Path): The file path to the env containing variables.

    Raises:
        FileNotFoundError: If .env file is not found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Strip optional surrounding quotes from the value
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)
