import re
import streamlit as st
import wikipedia
import ollama

# ── Page config (must be first Streamlit call) ───────────────────────────────
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

# ── Constants ─────────────────────────────────────────────────────────────────
NEED_WIKI      = "NEED_WIKI"
AVAILABLE_MODELS = ["qwen2.5:3b", "phi3:mini", "tinyllama:latest"]


# ── Core logic (from local_wiki_chatbot.py) ───────────────────────────────────

def clean_query(query: str) -> str:
    prefixes = [
        r"^tell me about\s+", r"^what is\s+", r"^what are\s+",
        r"^who is\s+",        r"^who was\s+", r"^explain\s+",
        r"^describe\s+",      r"^give me information (on|about)\s+",
        r"^i want to know about\s+", r"^can you tell me about\s+",
        r"^do you know about\s+",    r"^search for\s+", r"^look up\s+",
    ]
    cleaned = query.strip()
    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or query


def search_wikipedia(query: str, sentences: int = 10):
    try:
        search_term = clean_query(query)
        results = wikipedia.search(search_term, results=3)
        if not results:
            return None, "No Wikipedia results found."
        page    = wikipedia.page(results[0], auto_suggest=False)
        summary = wikipedia.summary(results[0], sentences=sentences, auto_suggest=False)
        return page.title, summary
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page    = wikipedia.page(e.options[0], auto_suggest=False)
            summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
            return page.title, summary
        except Exception:
            return None, f"Ambiguous topic. Try: {', '.join(e.options[:4])}"
    except wikipedia.exceptions.PageError:
        return None, f"No Wikipedia page found for '{query}'."
    except Exception as ex:
        return None, f"Wikipedia error: {ex}"


def ask_model_direct(user_query: str, model: str) -> str:
    system_prompt = (
        "You are a knowledgeable assistant. "
        "Answer the user's question clearly and concisely using your own knowledge. "
        "However, if you are NOT confident, or the topic is too specific, niche, or recent, "
        "respond with ONLY the word: NEED_WIKI (nothing else). "
        "Do NOT use NEED_WIKI if you genuinely know the answer."
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ]
    )
    return response["message"]["content"]


def ask_model_with_context(user_query: str, wiki_title: str, wiki_context: str, model: str) -> str:
    system_prompt = (
        "You are a knowledgeable assistant. "
        "You will be given a Wikipedia article as context. "
        "Use ONLY that context to answer the user's question with a clear, concise summary. "
        "Do not fabricate information beyond what the context provides."
    )
    user_message = (
        f"Wikipedia article: '{wiki_title}'\n\n"
        f"--- CONTEXT START ---\n{wiki_context}\n--- CONTEXT END ---\n\n"
        f"User question: {user_query}\n\n"
        f"Please provide a helpful and concise summary based on the context above."
    )
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
    )
    return response["message"]["content"]


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
                    reply_placeholder.markdown(f"⚠️ {wiki_context}\n\nSorry, I couldn't find enough information to answer that.")
                    badge_placeholder.markdown('<span class="badge-none">📌 No source found</span>', unsafe_allow_html=True)
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
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "source": "none"
            })
