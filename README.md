# Relay 🔁

> A minimal, typed Python unified interface for native LLMs.

One request schema. One response schema. Swap providers without touching your application code.

![alt text](docs/relay_st.gif)

Implementations are done via both REST (from scratch) and SDK (direct) for learning purposes.

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

## Usage

### Basic generation

```python
# Imports
from relay.llm.factory import LlmProviderFactory
from relay.llm.schemas import LlmRequest, LlmMessage, Role
# Arrange (creation)
llm = LlmProviderFactory.create(
    provider_type="google",
    api_key="sk-ant-...",
    model_name="gemini-2.5-flash",     
    implementation="sdk",              
)
request = LlmRequest(
    messages=[LlmMessage(role=Role.user, content="Explain transformers in one paragraph.")],
    temperature=0.7,
)
# Act (generation)
response = await llm.generate(request)
print(response.content)
```

### Listing available models

```python
# Create the LLM without a model name, fetch the list, then set it.
llm = LlmProviderFactory.create("google", api_key="sk-...")
models = await llm.list_models()
print(models)
```

### Streaming

```python
async for chunk in await llm.generate(request, stream=True):
    print(chunk, end="", flush=True)
```

### System prompts

```python
request = LlmRequest(
    messages=[LlmMessage(role=Role.user, content="Summarise this.")],
    system_prompt="You are a concise technical writer.",
)
```

### Switching providers

```python
# Same request, different provider - no other changes needed
llm = LlmProviderFactory.create("google", api_key="AIza...", model_name="gemini-2.5-flash")
response = await llm.generate(request)
```

---

## Interfaces

Relay ships with two ready-made interfaces for interacting with any provider directly.

**CLI**
```bash
python -m relay.cli
```

**Streamlit app**
```bash
streamlit run relay/app.py
```

Both prompt for provider, implementation (sdk/rest), API key, and let you pick from a live model list - nothing hardcoded, nothing stored.

---

## Supported methods

| Version | Non-streaming | Streaming | System prompt | Thinking Mode | Tool/Function Calling | Image Generation | Voice Generation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| v1 | ✓ | ✓ | ✓ | | | | |
| v2 | | | | soon | | | |
| v3 | | | | | soon | | |
| v4 | | | | | | soon | |
| v5 | | | | | | | soon |

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