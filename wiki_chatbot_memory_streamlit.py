import streamlit as st

# ── Import all model logic from the main module ───────────────────────────────
from wiki_chatbot_memory import (
    NEED_WIKI,
    resolve_topic,
    search_wikipedia,
    build_history_for_model,
    stream_model_direct,
    stream_model_with_context,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Big Pickle Chat",
    page_icon="🥒",
    layout="wide",
)

# ── Black theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stMain"],
.main { background-color: #080808 !important; color: #e0e0e0 !important; font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0d0d0d !important; border-right: 1px solid #1e1e1e; }
[data-testid="stSidebar"] * { color: #b0b0b0 !important; font-family: 'Inter', sans-serif; }

/* Sidebar header */
[data-testid="stSidebar"] h2 { color: #ffffff !important; font-size: 1rem !important; }

/* Input */
.stChatInputContainer textarea,
[data-testid="stChatInput"] textarea {
    background-color: #111111 !important;
    color: #f0f0f0 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #0e0e0e !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    padding: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] code {
    color: #dcdcdc !important;
}

/* Code blocks */
[data-testid="stChatMessage"] pre {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
}

/* Source badges */
.badge-model {
    display: inline-block;
    background: #0d2010;
    border: 1px solid #1a5c2a;
    color: #4caf7d;
    font-size: 0.70rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-top: 10px;
    font-weight: 600;
    letter-spacing: 0.4px;
    font-family: 'Inter', sans-serif;
}
.badge-wiki {
    display: inline-block;
    background: #0d1a2e;
    border: 1px solid #1a3a6c;
    color: #5b9bd5;
    font-size: 0.70rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-top: 10px;
    font-weight: 600;
    letter-spacing: 0.4px;
    font-family: 'Inter', sans-serif;
}
.badge-none {
    display: inline-block;
    background: #1c0e0e;
    border: 1px solid #5c1a1a;
    color: #e05555;
    font-size: 0.70rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-top: 10px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
}

/* Status text */
.status-step {
    color: #666;
    font-size: 0.82rem;
    font-style: italic;
    font-family: 'Inter', sans-serif;
    padding: 4px 0;
}

/* Title */
.main-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.8px;
    margin-bottom: 0;
    font-family: 'Inter', sans-serif;
}
.main-subtitle {
    font-size: 0.84rem;
    color: #555;
    margin-top: 3px;
    margin-bottom: 18px;
    font-family: 'Inter', sans-serif;
}

/* Buttons */
.stButton > button {
    background: #141414;
    color: #cccccc;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #1e1e1e;
    border-color: #444;
    color: #ffffff;
}

/* Selectbox */
[data-baseweb="select"] { background-color: #111 !important; border-color: #2a2a2a !important; }
[data-baseweb="select"] * { color: #e0e0e0 !important; background-color: #111 !important; }

hr { border-color: #1e1e1e !important; }

/* Hide the Streamlit top bar */
[data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # Each: {role, content, source, wiki_title (optional)}
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥒 Big Pickle Chat")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
1. 🧠 Answers from own knowledge  
2. 🔎 Resolves "it / that / more" → real topic  
3. 📖 Wikipedia fallback if unsure  
4. 💬 Streams tokens live (no blank screen!)  
5. 📌 Source shown on every reply  
""")
    st.markdown("---")
    st.markdown(f"**Session tokens used:** `{st.session_state.total_tokens:,}`")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<span style='color:#333;font-size:0.72rem;'>Memory-enabled · Streaming · Wikipedia fallback</span>",
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🥒 Big Pickle Chat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">Memory-enabled · Powered by Big Pickle + Wikipedia fallback · Streams live</div>',
    unsafe_allow_html=True
)
st.markdown("---")


# ── Render existing chat history ──────────────────────────────────────────────
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
user_input = st.chat_input("Ask me anything... (memory is ON 🧠)", key="chat_input")

if user_input:
    # 1) Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Build model-ready history (all previous turns)
    history = build_history_for_model(st.session_state.messages)

    # 3) Generate assistant response
    with st.chat_message("assistant"):
        status_area = st.empty()
        reply_area  = st.empty()
        badge_area  = st.empty()

        try:
            # ── Step 1: Stream direct model response ─────────────────────────
            status_area.markdown(
                '<span class="status-step">🧠 Thinking with conversation memory...</span>',
                unsafe_allow_html=True
            )

            full_reply = ""
            stream_box = reply_area.empty()

            for token in stream_model_direct(history):
                full_reply += token
                # Buffer first 10 chars so we don't display a partial "NEED_WIKI"
                if len(full_reply) > 10:
                    stream_box.markdown(full_reply + "▌")

            # Final render without cursor
            if full_reply.strip() != NEED_WIKI:
                stream_box.markdown(full_reply)

            # Track tokens (stored on the generator function after iteration)
            usage = stream_model_direct._last_usage
            if usage:
                st.session_state.total_tokens += usage.total_tokens

            # ── Step 2: Was it NEED_WIKI? ────────────────────────────────────
            if full_reply.strip() == NEED_WIKI:
                stream_box.empty()

                status_area.markdown(
                    '<span class="status-step">🔎 Resolving topic from conversation context...</span>',
                    unsafe_allow_html=True
                )
                resolved_topic = resolve_topic(user_input, st.session_state.messages[:-1])

                status_area.markdown(
                    f'<span class="status-step">🔍 Searching Wikipedia for <b>"{resolved_topic}"</b>...</span>',
                    unsafe_allow_html=True
                )
                wiki_title, wiki_context = search_wikipedia(resolved_topic)

                if wiki_title is None:
                    status_area.empty()
                    err_msg = f"⚠️ {wiki_context}\n\nSorry, I couldn't find information on that."
                    reply_area.markdown(err_msg)
                    badge_area.markdown('<span class="badge-none">📌 No source found</span>', unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg, "source": "none"}
                    )
                else:
                    status_area.markdown(
                        f'<span class="status-step">📖 Found <b>"{wiki_title}"</b> · Generating answer...</span>',
                        unsafe_allow_html=True
                    )

                    wiki_reply = ""
                    wiki_stream_box = reply_area.empty()
                    for token in stream_model_with_context(history, wiki_title, wiki_context):
                        wiki_reply += token
                        wiki_stream_box.markdown(wiki_reply + "▌")
                    wiki_stream_box.markdown(wiki_reply)

                    usage2 = stream_model_with_context._last_usage
                    if usage2:
                        st.session_state.total_tokens += usage2.total_tokens

                    status_area.empty()
                    badge_area.markdown(
                        f'<span class="badge-wiki">📌 Wikipedia — {wiki_title}</span>',
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": wiki_reply,
                        "source": "wiki",
                        "wiki_title": wiki_title,
                    })

            else:
                # ── Model answered confidently ────────────────────────────────
                status_area.empty()
                badge_area.markdown(
                    '<span class="badge-model">📌 Model\'s own knowledge</span>',
                    unsafe_allow_html=True
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_reply,
                    "source": "model",
                })

        except Exception as e:
            status_area.empty()
            err_msg = f"❌ **Error:** {e}"
            reply_area.markdown(err_msg)
            badge_area.markdown('<span class="badge-none">📌 Error</span>', unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": err_msg, "source": "none"}
            )

    # Refresh sidebar token count
    st.rerun()
