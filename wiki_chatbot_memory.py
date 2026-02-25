import os
import re
import wikipedia
from dotenv import load_dotenv
from openai import OpenAI

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()

api_key = os.getenv("OPENCODE_API_KEY")
if not api_key:
    print("Error: OPENCODE_API_KEY not found. Make sure it's in your .env file.")
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://opencode.ai/zen/v1"
)

MODEL     = "big-pickle"
NEED_WIKI = "NEED_WIKI"

# ── Conversation memory (used by console chatbot) ─────────────────────────────
# Each entry: {"role": "user"|"assistant", "content": "..."}
conversation_history = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_query(query: str) -> str:
    """Strip conversational filler so Wikipedia gets a clean topic."""
    prefixes = [
        r"^tell me about\s+",
        r"^what is\s+",
        r"^what are\s+",
        r"^who is\s+",
        r"^who was\s+",
        r"^explain\s+",
        r"^describe\s+",
        r"^give me information (on|about)\s+",
        r"^i want to know about\s+",
        r"^can you tell me about\s+",
        r"^do you know about\s+",
        r"^search for\s+",
        r"^look up\s+",
    ]
    cleaned = query.strip()
    for pattern in prefixes:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or query


def resolve_topic(user_query: str, history: list = None) -> str:
    """
    Use the conversation history to resolve vague references like 'it', 'that',
    'the element', 'tell me more', etc. into a concrete Wikipedia search topic.

    Args:
        user_query: The user's latest message.
        history: Optional list of {"role", "content"} dicts. Defaults to the
                 module-level conversation_history if not provided.

    Returns the resolved topic string (e.g. 'Atom (physics)').
    """
    hist = history if history is not None else conversation_history

    if not hist:
        return clean_query(user_query)

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:300]}"
        for m in hist[-6:]
    )

    resolver_prompt = (
        "You are a search query resolver. "
        "Given a conversation history and the user's latest message, "
        "output ONLY a short, clear Wikipedia search query that captures what the user is asking about. "
        "Resolve pronouns like 'it', 'that', 'the element', 'tell me more' to the actual topic. "
        "Do NOT explain. Output ONLY the search query — nothing else.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User's latest message: {user_query}\n\n"
        "Wikipedia search query:"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": resolver_prompt}],
    )
    resolved = resp.choices[0].message.content.strip().strip('"\'')
    return resolved if resolved else clean_query(user_query)


def search_wikipedia(search_term: str, sentences: int = 10):
    """Search Wikipedia. Returns (title, summary) or (None, error_message)."""
    try:
        results = wikipedia.search(search_term, results=3)
        if not results:
            return None, "No Wikipedia results found for your query."

        page    = wikipedia.page(results[0], auto_suggest=False)
        summary = wikipedia.summary(results[0], sentences=sentences, auto_suggest=False)
        return page.title, summary

    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page    = wikipedia.page(e.options[0], auto_suggest=False)
            summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
            return page.title, summary
        except Exception:
            return None, f"Topic is ambiguous. Options include: {', '.join(e.options[:5])}"

    except wikipedia.exceptions.PageError:
        return None, f"No Wikipedia page found for '{search_term}'."

    except Exception as ex:
        return None, f"Wikipedia search error: {ex}"


def ask_model_direct(history: list = None) -> tuple:
    """
    Send conversation history to the model.
    The system prompt tells it to reply NEED_WIKI if it's not confident.

    Args:
        history: Optional list of {"role", "content"} dicts. Defaults to the
                 module-level conversation_history if not provided.

    Returns (reply_text, usage, headers).
    """
    hist = history if history is not None else conversation_history

    system_prompt = (
        "You are a knowledgeable assistant with memory of the full conversation. "
        "Answer the user's latest question clearly and concisely using your own knowledge. "
        "You have access to everything said earlier in the conversation — use it! "
        "However, if you are NOT confident in your answer or the topic is too specific, "
        "niche, or recent for you to answer accurately, respond with ONLY the word: NEED_WIKI "
        "(nothing else, no explanation). Do NOT use NEED_WIKI if you genuinely know the answer."
    )

    messages = [{"role": "system", "content": system_prompt}] + hist

    raw = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=messages,
    )
    response = raw.parse()
    return response.choices[0].message.content, response.usage, raw.headers


def ask_model_with_context(wiki_title: str, wiki_context: str, history: list = None) -> tuple:
    """
    Send conversation history PLUS Wikipedia context to the model.

    Args:
        wiki_title: The Wikipedia article title.
        wiki_context: The Wikipedia article summary text.
        history: Optional list of {"role", "content"} dicts. Defaults to the
                 module-level conversation_history if not provided.

    Returns (reply_text, usage, headers).
    """
    hist = history if history is not None else conversation_history

    system_prompt = (
        "You are a knowledgeable assistant with memory of the full conversation. "
        "You are given a Wikipedia article as extra context. "
        "Use the conversation history AND the Wikipedia context to answer the user's latest question "
        "with a clear, concise summary. Do not fabricate information beyond what is provided."
    )

    wiki_note = (
        f"[Wikipedia article fetched for this query: '{wiki_title}']\n"
        f"--- CONTEXT START ---\n{wiki_context}\n--- CONTEXT END ---"
    )

    messages = (
        [{"role": "system", "content": system_prompt}]
        + [{"role": "system", "content": wiki_note}]
        + hist
    )

    raw = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=messages,
    )
    response = raw.parse()
    return response.choices[0].message.content, response.usage, raw.headers


def stream_model_direct(history: list):
    """
    Stream the model response using conversation history.
    System prompt includes the NEED_WIKI instruction.
    Yields text chunks.
    """
    system_prompt = (
        "You are a knowledgeable assistant with memory of the full conversation. "
        "Answer the user's latest question clearly and concisely using your own knowledge. "
        "You have access to everything said earlier — use it! "
        "However, if you are NOT confident or the topic is too specific / niche / recent, "
        "respond with ONLY the word: NEED_WIKI (nothing else). "
        "Do NOT use NEED_WIKI if you genuinely know the answer."
    )
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    last_usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if hasattr(chunk, "usage") and chunk.usage:
            last_usage = chunk.usage
    # Return usage via a special attribute on the generator — callers can access
    # it via the module-level variable set below after iteration.
    stream_model_direct._last_usage = last_usage


def stream_model_with_context(history: list, wiki_title: str, wiki_context: str):
    """
    Stream the model response with Wikipedia context injected.
    Yields text chunks.
    """
    system_prompt = (
        "You are a knowledgeable assistant with memory of the full conversation. "
        "A Wikipedia article has been fetched as context. "
        "Use the conversation history AND the Wikipedia context to answer the user's latest question "
        "with a clear, concise summary. Do not fabricate information beyond what is provided."
    )
    wiki_note = (
        f"[Wikipedia article fetched: '{wiki_title}']\n"
        f"--- CONTEXT START ---\n{wiki_context}\n--- CONTEXT END ---"
    )
    messages = (
        [{"role": "system", "content": system_prompt}]
        + [{"role": "system", "content": wiki_note}]
        + history
    )
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    last_usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if hasattr(chunk, "usage") and chunk.usage:
            last_usage = chunk.usage
    stream_model_with_context._last_usage = last_usage


def build_history_for_model(messages: list) -> list:
    """Convert session messages (which may have extra keys) to role/content list."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


# ── Main chatbot loop (console) ───────────────────────────────────────────────

def start_chatbot():
    print("🤖 Smart Chatbot with Memory  (Big Pickle + Wikipedia fallback)")
    print("   I remember everything said in this session!")
    print("   Feel free to say 'tell me more', 'what about its history', etc.")
    print("   Type 'exit' or 'quit' to stop.\n")
    print("-" * 60)

    total_session_tokens = 0

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print(f"\n📊 Session Total Tokens Used: {total_session_tokens}")
            print("Goodbye!")
            break

        # Add user message to history BEFORE calling the model
        conversation_history.append({"role": "user", "content": user_input})

        try:
            # ── Step 1: Ask the model with full conversation history ──────────
            print("\n🧠 Thinking (with memory of our conversation)...")
            reply, usage, headers = ask_model_direct()
            total_session_tokens += usage.total_tokens

            if reply.strip() == NEED_WIKI:
                # ── Step 2: Resolve the topic using conversation context ──────
                print("🔎 Model unsure — resolving topic from conversation context...")
                resolved_topic = resolve_topic(user_input)
                print(f"🔍 Searching Wikipedia for: '{resolved_topic}'...")

                wiki_title, wiki_context = search_wikipedia(resolved_topic)

                if wiki_title is None:
                    print(f"⚠️  {wiki_context}")
                    bot_reply = "Sorry, I don't have enough information to answer that."
                    print(f"Big Pickle: {bot_reply}")
                    print("📌 Source: Neither model nor Wikipedia could answer")
                else:
                    print(f"✅ Found Wikipedia article: '{wiki_title}'")
                    print("💬 Generating answer from Wikipedia context...\n")
                    reply, usage2, headers = ask_model_with_context(wiki_title, wiki_context)
                    total_session_tokens += usage2.total_tokens
                    usage = usage2
                    bot_reply = reply
                    print(f"Big Pickle: {bot_reply}")
                    print(f"📌 Source: Answered using Wikipedia — '{wiki_title}'")

            else:
                # ── Model answered confidently ────────────────────────────────
                bot_reply = reply
                print(f"Big Pickle: {bot_reply}")
                print("📌 Source: Answered from model's own knowledge")

            # Add assistant reply to history so next turn has full context
            conversation_history.append({"role": "assistant", "content": bot_reply})

            tokens_remaining = headers.get("x-ratelimit-remaining-tokens", "N/A")
            tokens_limit     = headers.get("x-ratelimit-limit-tokens", "N/A")

            print(
                f"\n📊 Tokens — Prompt: {usage.prompt_tokens} | "
                f"Completion: {usage.completion_tokens} | "
                f"Total this call: {usage.total_tokens} | "
                f"Session total: {total_session_tokens}"
            )
            print(f"🔋 Tokens Remaining: {tokens_remaining} / {tokens_limit}")

        except Exception as e:
            # Remove the user message from history on error so it can be retried
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
            print(f"❌ Error: {e}")

        print("-" * 60)


if __name__ == "__main__":
    start_chatbot()
