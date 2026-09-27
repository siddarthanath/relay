# Relay 🔁

> A minimal unified Python interface for LLMs.

One request schema. One response schema. Swap providers without touching your application code.


## Why

- Building with LLMs always ends up with the same problem - every provider has a different SDK, a different message format, a different streaming interface. Switching from OpenAI to Anthropic means rewriting your entire LLM layer.
- Relay standardises this into one request schema and one response schema. Swap providers by changing one string. The rest of your code doesn't move.

Providers are implemented on top of their official **SDKs**, wrapped behind a single typed interface.

![alt text](docs/relay_st.gif)

---

## Install

### Approach 1: GitHub
```bash
git clone https://github.com/siddarthanath/relay
cd relay
pip install -e .
```

### Approach 2: PyPI
```bash
pip install relay
```

---

## Architecture

```mermaid
flowchart TD
    User(["User / App"])

    subgraph Interfaces
        CLI["CLI"]
        ST["Streamlit App"]
    end

    subgraph Schemas
        REQ["LlmRequest"]
        RESP["LlmResponse"]
    end

    Factory["LlmProviderFactory"]

    subgraph Base["BaseLlm"]
        GEN["generate()"]
        LIST["list_models()"]
    end

    subgraph Providers["Providers"]
        P_A["AnthropicLlm"]
        P_O["OpenAILlm"]
        P_G["GoogleLlm"]
    end

    LLMs["External LLM APIs"]

    User --> CLI
    User --> ST
    CLI --> Factory
    ST --> Factory
    User --> Factory
    Factory --> Base
    Base --> Providers
    Providers --> P_A
    Providers --> P_O
    Providers --> P_G
    P_A --> LLMs
    P_O --> LLMs
    P_G --> LLMs
    LLMs --> Base
    Base --> User
    REQ -.-> GEN
    GEN -.-> RESP
```

---

## Usage

### 1a. Basic generation

Use the **factory** when you supply the API key yourself at call time:

```python
# Imports
from relay.llm import LlmProviderFactory
from relay.llm.schemas import LlmRequest, LlmMessage, Role
# Arrange (LLM creation)
llm = LlmProviderFactory.create(provider="google",
                                api_key="AIza...",
                                model_name="gemini-2.5-flash")
request = LlmRequest(messages=[LlmMessage(role=Role.user, 
                                          content="Explain transformers in one paragraph.")
                              ],
                     temperature=0.7)
# Act (LLM generation)
response = await llm.generate(request)
print(response.content)
```

Use the **registry** when keys live in a `.env` file - configure once, fetch anywhere:

```python
# Imports
from relay.llm import LlmProviderRegistry
from relay.llm.schemas import LlmRequest, LlmMessage, Role
# Arrange (LLM creation)
registry = LlmProviderRegistry(env_file=".env")
llm = registry.get("google")
request  = LlmRequest(messages=[LlmMessage(role=Role.user, 
                                           content="Explain transformers in one paragraph.")
                               ],
                      temperature=0.7)
# Act (LLM generation)
response = await llm.generate(request)
print(response.content)
```

### 1b. Context manager

Use `async with` to ensure the underlying HTTP client is closed when you're done:

```python
# Imports
from relay.llm import LlmProviderFactory
from relay.llm.schemas import LlmRequest, LlmMessage, Role
# Arrange (LLM creation)
llm = LlmProviderFactory.create(provider="openai",
                                api_key="sk-...",
                                model_name="gpt-4o")
request = LlmRequest(messages=[LlmMessage(role=Role.user,
                                          content="Explain transformers in one paragraph.")
                              ],
                     temperature=0.7)
# Act (LLM generation) — client is closed automatically on exit
async with llm:
    response = await llm.generate(request)
    print(response.content)
```


### 2. Listing available models

```python
llm = LlmProviderFactory.create(provider="google", 
                                api_key="AIza...")
models = await llm.list_models()
print(models)
```

### 3. Streaming

```python
async for chunk in await llm.generate(request, stream=True):
    print(chunk, end="", flush=True)
```

### 4. System prompts

```python
request = LlmRequest(messages=[LlmMessage(role=Role.user, 
                                          content="Summarise this.")
                              ],
                     system_prompt="You are a concise technical writer.")
```

### 5. Switching providers

```python
# Same request, different provider — no other changes needed
llm = LlmProviderFactory.create(provider="anthropic", 
                                api_key="sk-ant-...", 
                                model_name="claude-sonnet-4-20250514")
response = await llm.generate(request)
```

---

## Interfaces

Relay ships with two ready-made example interfaces (in `examples/`) for interacting with any provider directly.

**CLI**
```bash
python examples/cli.py
```

**Streamlit app**
```bash
streamlit run examples/streamlit_app.py
```

Both prompt for provider and API key at launch - nothing hardcoded, nothing stored. The Streamlit app pulls a live model list from the provider so you always see what's available.

> The CLI needs the `cli` extra (`pip install -e ".[cli]"`) and the Streamlit app needs the `ui` extra (`pip install -e ".[ui]"`). Both are included in `all`.

---

## Roadmap

| Version | Feature | Status |
|:---:|:---| :---|
| v1 | Non-streaming, streaming, system prompts | ✓
| v2 | Thinking mode (o1, Claude extended thinking) | ✗
| v3 | Tool and function calling | ✗
| v4 | Image generation | ✗
| v5 | Voice generation | ✗

---

## Citation

If you use Relay in your work, please cite:

```text
@software{relay2026,
  author = {Siddartha Nath},
  title = {Relay: A Minimal Unified Python Interface for LLMs},
  year = {2026},
  url = {https://github.com/siddarthanath/relay}
}
```