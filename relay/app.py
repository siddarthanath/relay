# ───────────────────────────────────────────────────── Imports ────────────────────────────────────────────────────── #

# Standard Library
import asyncio
import queue
import threading

# Third Party Library
import streamlit as st

# Private Library
from relay.llm.factory import LlmProviderFactory
from relay.llm.schemas import LlmMessage, LlmRequest, Role
from relay.llm.constants import _DEFAULT_MODELS

# ────────────────────────────────────────────────────── Code ──────────────────────────────────────────────────────── #

def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop stored in session state."""
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
    return st.session_state.event_loop

def _stream_response(llm, request: LlmRequest):
    """Bridge async streaming to a sync generator for st.write_stream."""
    q: queue.Queue = queue.Queue()
    loop = _get_loop()

    def _run() -> None:
        asyncio.set_event_loop(loop)

        async def _async_stream() -> None:
            async for chunk in await llm.generate(request, stream=True):
                q.put(chunk)
            q.put(None)

        loop.run_until_complete(_async_stream())

    threading.Thread(target=_run, daemon=True).start()

    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk

def _init_state() -> None:
    defaults = {"history": [], "llm": None, "model_name": None, "event_loop": None}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state.event_loop is None:
        st.session_state.event_loop = asyncio.new_event_loop()

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        ```
        ██████╗ ███████╗██╗      █████╗ ██╗   ██╗
        ██╔══██╗██╔════╝██║     ██╔══██╗╚██╗ ██╔╝
        ██████╔╝█████╗  ██║     ███████║ ╚████╔╝
        ██╔══██╗██╔══╝  ██║     ██╔══██║  ╚██╔╝
        ██║  ██║███████╗███████╗██║  ██║   ██║
        ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝
        ```
        """)
        st.caption("A minimal, typed Python unified interface for native LLM SDKs.")
        st.divider()

        st.subheader("Configuration")

        st.selectbox("Interface Provider", ["native"], key="provider")

        model = st.selectbox("Model Family", ["anthropic", "google", "openai"], key="model_family")

        st.text_input("Model Version", value=_DEFAULT_MODELS[model], key="model_version")

        st.text_input("API Key", type="password", placeholder=f"{model.capitalize()} API key", key="api_key")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", type="primary", use_container_width=True):
                _handle_connect()
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        if st.session_state.llm:
            st.success(f"Connected: **{st.session_state.model_name}**")

def _handle_connect() -> None:
    api_key    = st.session_state.get("api_key", "").strip()
    model      = st.session_state.get("model_family", "google")
    model_name = st.session_state.get("model_version", _DEFAULT_MODELS[model]).strip()

    if not api_key:
        st.sidebar.error("API key is required.")
        return

    with st.sidebar:
        with st.spinner("Connecting..."):
            try:
                st.session_state.llm        = LlmProviderFactory.create(model, api_key, model_name)
                st.session_state.model_name = model_name
            except Exception as e:
                st.error(f"Failed to connect: {e}")

def main() -> None:
    st.set_page_config(page_title="Relay", page_icon="🔁", layout="wide")
    st.markdown("<style>[data-testid='stSidebar']{min-width:475px}</style>", unsafe_allow_html=True)
    _init_state()
    _sidebar()

    st.title("🔁 Relay Chat")
    st.caption("Configure your model in the sidebar, then start chatting.")
    st.divider()

    if not st.session_state.llm:
        st.info("Connect a model using the sidebar to get started.")
        return

    # Render chat history
    for msg in st.session_state.history:
        role = "user" if msg.role == Role.user else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Input
    if prompt := st.chat_input("Message..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.history.append(LlmMessage(role=Role.user, content=prompt))
        request = LlmRequest(messages=st.session_state.history)

        with st.chat_message("assistant"):
            try:
                response = st.write_stream(_stream_response(st.session_state.llm, request))
            except Exception as e:
                st.session_state.history.pop()
                st.error(f"Error: {e}")
                return

        st.session_state.history.append(LlmMessage(role=Role.assistant, content=response))

if __name__ == "__main__":
    main()
