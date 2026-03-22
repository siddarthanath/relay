# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
from datetime import datetime
import textwrap

# Third Party Library
import click

# Private Library
from relay.llm.factory import LlmProviderFactory
from relay.llm.schemas import LlmMessage, LlmRequest, Role
from relay.llm.constants import _PROVIDERS, _IMPLEMENTATIONS, _MODELS

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

BANNER = r"""
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║   ██████╗ ███████╗██╗      █████╗ ██╗   ██╗               ║
  ║   ██╔══██╗██╔════╝██║     ██╔══██╗╚██╗ ██╔╝               ║
  ║   ██████╔╝█████╗  ██║     ███████║ ╚████╔╝                ║
  ║   ██╔══██╗██╔══╝  ██║     ██╔══██║  ╚██╔╝                 ║
  ║   ██║  ██║███████╗███████╗██║  ██║   ██║                  ║
  ║   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝                  ║
  ║                                                           ║
  ║         A minimal, typed Python unified interface         ║
  ║                  for native LLM SDKs.                     ║
  ╚═══════════════════════════════════════════════════════════╝
"""

WIDTH   = 62
DIVIDER = "─" * WIDTH

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S (%Y-%m)")

def _chat_header(model_name: str) -> str:
    label = f"  Chat  │ Model [{model_name}]  "
    right_pad = "─" * max(0, WIDTH - len(label))
    return f"┌{'─' * WIDTH}┐\n│{label}{right_pad}│\n└{'─' * WIDTH}┘"

def _render_box(sender: str, content: str, ts: str) -> str:
    prefix    = "  >  "
    indent    = "     "
    inner     = WIDTH - 2
    text_w    = inner - len(prefix)
    header    = f" {ts}  {sender}"

    wrapped = textwrap.wrap(content, width=text_w) or [""]
    body_lines = [f"│ {prefix}{wrapped[0]:<{inner - len(prefix)}} │"]
    for line in wrapped[1:]:
        body_lines.append(f"│ {indent}{line:<{inner - len(indent)}} │")

    top    = f"┌{'─' * WIDTH}┐"
    hdr    = f"│ {header:<{inner}} │"
    sep    = f"├{'─' * WIDTH}┤"
    bottom = f"└{'─' * WIDTH}┘"

    return "\n".join([top, hdr, sep] + body_lines + [bottom])

def _echo(msg: str = "") -> None:
    click.echo(msg)

def _header(msg: str) -> None:
    _echo()
    _echo(f"  {msg}")
    _echo(DIVIDER)

def _prompt(label: str, choices: list[str] | None = None) -> str:
    hint = f" ({', '.join(choices)})" if choices else ""
    while True:
        value = click.prompt(f"  {label}{hint}").strip()
        if not value:
            _echo("  Value cannot be empty. Try again.")
            continue
        if choices and value.lower() not in choices:
            _echo(f"  Invalid choice. Pick one of: {', '.join(choices)}")
            continue
        return value.lower() if choices else value

def _pick_model(llm) -> str:
    """Fetch available models and prompt the user to pick one by number."""
    _echo("  Fetching available models...")
    try:
        models: list[str] = asyncio.run(llm.list_models())
    except Exception as e:
        _echo(f"  Could not fetch model list: {e}")
        _echo("  Enter model name manually:")
        return input("  Model name > ").strip()

    _echo()
    for i, name in enumerate(models, 1):
        _echo(f"    {i:>3}.  {name}")
    _echo()

    while True:
        raw = click.prompt("  Pick a number").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        _echo(f"  Enter a number between 1 and {len(models)}.")

def _setup():
    """Interactive setup sequence. Returns (llm, model_name)."""

    _header("Step 1 of 4  —  Interface Provider")
    _echo("  How do you want to connect to the model?")
    _echo()
    _echo("    native     Direct connection (no middleware)")
    _echo("    langchain  LangChain abstraction layer  [coming soon]")
    _echo()
    _prompt("Provider", choices=_PROVIDERS)  # only "native" supported

    _header("Step 2 of 4  —  Implementation")
    _echo("  Which backend should be used to call the API?")
    _echo()
    _echo("    sdk   Official provider SDK  (recommended)")
    _echo("    rest  Raw HTTP via httpx")
    _echo()
    impl = _prompt("Implementation", choices=_IMPLEMENTATIONS)

    _header("Step 3 of 4  —  Model Family")
    _echo("  Which model family do you want to use?")
    _echo()
    _echo("    anthropic  Claude (Sonnet, Opus, Haiku)")
    _echo("    google     Gemini (Flash, Pro)")
    _echo("    openai     GPT (4o, o1, o3)")
    _echo()
    model = _prompt("Model", choices=_MODELS)

    _header("Step 4 of 4  —  API Key")
    _echo(f"  Enter your {model.capitalize()} API key.")
    _echo()
    api_key = input("  API key > ").strip()

    _echo()
    _echo(DIVIDER)

    # Create a temporary LLM (no model name yet) purely to fetch the model list.
    try:
        temp_llm = LlmProviderFactory.create(model, api_key, implementation=impl)
    except Exception as e:
        _echo(f"\n  Failed to initialise client: {e}")
        raise SystemExit(1)

    _header("Step 5 of 5  —  Model Version")
    model_name = _pick_model(temp_llm)

    _echo()
    _echo(DIVIDER)
    _echo("  Connecting...")

    try:
        llm = LlmProviderFactory.create(model, api_key, model_name, implementation=impl)
    except Exception as e:
        _echo(f"\n  Failed to initialise client: {e}")
        raise SystemExit(1)

    _echo(f"  Connected.  [{model.capitalize()} / {model_name}]")
    _echo(DIVIDER)

    return llm, model_name

async def _chat_loop(llm, model_name: str) -> None:
    history: list[LlmMessage] = []

    def _render_header() -> None:
        click.clear()
        _echo(_chat_header(model_name))
        _echo("  'clear' to reset  │  'exit' to quit")
        _echo(DIVIDER)

    _render_header()

    while True:
        try:
            user_input = click.prompt("\n  You >", prompt_suffix=" ")
        except (KeyboardInterrupt, EOFError):
            _echo("Thank you for using the CLI chat interface. See you again soon!")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.lower() in ("exit", "quit"):
            _echo("Thank you for using the CLI chat interface. See you again soon!")
            break

        if stripped.lower() == "clear":
            history.clear()
            _render_header()
            continue

        _echo(_render_box("You", stripped, _ts()))

        history.append(LlmMessage(role=Role.user, content=stripped))
        request = LlmRequest(messages=history)

        try:
            chunks = []
            async for chunk in await llm.generate(request, stream=True):
                chunks.append(chunk)
            full_response = "".join(chunks)
            _echo(_render_box("AI", full_response, _ts()))
        except Exception as e:
            import traceback
            history.pop()
            _echo(f"\n  Error: {e}\n")
            _echo(traceback.format_exc())
            continue

        history.append(LlmMessage(role=Role.assistant, content=full_response))

def main() -> None:
    click.clear()
    _echo(BANNER)
    llm, model_name = _setup()
    asyncio.run(_chat_loop(llm, model_name))

if __name__ == "__main__":
    main()
