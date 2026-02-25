import streamlit as st

# ── Import all model logic from the main module ───────────────────────────────
from local_wiki_chatbot import (
    NEED_WIKI,
    AVAILABLE_MODELS,
    clean_query,
    search_wikipedia,
    ask_model_direct,
    ask_model_with_context,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Local Wiki Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Black theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #e0e0e0 !important;
    font-family: 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #1f1f1f;
}
[data-testid="stSidebar"] * { color: #cccccc !important; }

/* ── Input box ── */
.stChatInputContainer, [data-testid="stChatInput"] textarea {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
    padding: 12px !important;
}

/* ── User bubble ── */
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid #ffffff !important;
}

/* ── Assistant bubble ── */
[data-testid="stChatMessage"]:not([data-testid*="user"]) {
    border-left: 3px solid #555555 !important;
}

/* ── Source badges ── */
.badge-model {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    color: #00e676;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-wiki {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    color: #64b5f6;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-none {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    color: #ef5350;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 8px;
    font-weight: 600;
}

/* ── Divider ── */
hr { border-color: #1f1f1f !important; }

/* ── Buttons ── */
.stButton > button {
    background: #1a1a1a;
    color: #ffffff;
    border: 1px solid #333;
    border-radius: 8px;
}
.stButton > button:hover {
    background: #222222;
    border-color: #555;
}

/* ── Selectbox ── */
[data-baseweb="select"] {
    background-color: #111 !important;
    border-color: #333 !important;
    color: #fff !important;
}

/* ── Status text ── */
.status-text {
    color: #888888;
    font-size: 0.82rem;
    font-style: italic;
}

/* ── Title area ── */
.main-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.main-subtitle {
    font-size: 0.88rem;
    color: #666666;
    margin-top: 2px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, source, wiki_title}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    selected_model = st.selectbox(
        "🧠 Local Model",
        AVAILABLE_MODELS,
        index=0,
        help="Choose the Ollama model to use. Make sure `ollama serve` is running."
    )

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
1. 🧠 Model answers from own knowledge  
2. ❓ If unsure → searches Wikipedia  
3. 📖 Re-answers with Wikipedia context  
4. 📌 Shows source of every answer
""")
    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<span style='color:#444;font-size:0.75rem;'>Runs 100% locally · No API key needed</span>",
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🤖 Local Wiki Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="main-subtitle">Powered by <b>{selected_model}</b> · Wikipedia fallback enabled</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            source = msg.get("source", "model")
            if source == "model":
                st.markdown('<span class="badge-model">📌 Model\'s own knowledge</span>', unsafe_allow_html=True)
            elif source == "wiki":
                title = msg.get("wiki_title", "Wikipedia")
                st.markdown(f'<span class="badge-wiki">📌 Wikipedia — {title}</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-none">📌 No source found</span>', unsafe_allow_html=True)


# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...", key="chat_input")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        reply_placeholder  = st.empty()
        badge_placeholder  = st.empty()

        try:
            # Step 1: Ask model directly
            status_placeholder.markdown(
                f'<span class="status-text">🧠 Asking {selected_model} from its own knowledge...</span>',
                unsafe_allow_html=True
            )
            reply = ask_model_direct(user_input, selected_model)

            if reply.strip() == NEED_WIKI:
                # Step 2: Wikipedia fallback
                cleaned = clean_query(user_input)
                status_placeholder.markdown(
                    f'<span class="status-text">🔍 Model unsure — searching Wikipedia for "<b>{cleaned}</b>"...</span>',
                    unsafe_allow_html=True
                )
                wiki_title, wiki_context = search_wikipedia(user_input)

                if wiki_title is None:
                    status_placeholder.empty()
                    reply_placeholder.markdown(
                        f"⚠️ {wiki_context}\n\nSorry, I couldn't find enough information to answer that."
                    )
                    badge_placeholder.markdown(
                        '<span class="badge-none">📌 No source found</span>', unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ {wiki_context}\n\nSorry, I couldn't find enough information to answer that.",
                        "source": "none"
                    })
                else:
                    status_placeholder.markdown(
                        f'<span class="status-text">📖 Found "<b>{wiki_title}</b>" · Generating answer...</span>',
                        unsafe_allow_html=True
                    )
                    reply = ask_model_with_context(user_input, wiki_title, wiki_context, selected_model)
                    status_placeholder.empty()
                    reply_placeholder.markdown(reply)
                    badge_placeholder.markdown(
                        f'<span class="badge-wiki">📌 Wikipedia — {wiki_title}</span>',
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "source": "wiki",
                        "wiki_title": wiki_title
                    })

            else:
                # Model answered confidently
                status_placeholder.empty()
                reply_placeholder.markdown(reply)
                badge_placeholder.markdown(
                    '<span class="badge-model">📌 Model\'s own knowledge</span>',
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "source": "model"
                })

        except Exception as e:
            status_placeholder.empty()
            err_msg = f"❌ **Error:** {e}\n\nMake sure Ollama is running: `ollama serve`"
            reply_placeholder.markdown(err_msg)
            badge_placeholder.markdown(
                '<span class="badge-none">📌 Error</span>', unsafe_allow_html=True
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "source": "none"
            })
