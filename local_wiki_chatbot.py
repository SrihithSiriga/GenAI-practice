import re
import ollama
from langchain_community.retrievers import WikipediaRetriever

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME       = "qwen2.5:3b"
NEED_WIKI        = "NEED_WIKI"
AVAILABLE_MODELS = ["qwen2.5:3b", "phi3:mini", "tinyllama:latest"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_query(query: str) -> str:
    """
    Strip common conversational filler phrases so Wikipedia gets a clean topic.
    e.g. 'Tell me about element Mercury' -> 'element Mercury'
    """
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


def search_wikipedia(query: str, sentences: int = 10):
    """
    Search Wikipedia using LangChain's WikipediaRetriever.
    Returns (title, summary) or (None, error_message).
    """
    try:
        search_term = clean_query(query)
        retriever = WikipediaRetriever(top_k_results=1, doc_content_chars_max=sentences * 500)
        docs = retriever.invoke(search_term)
        if not docs:
            return None, "No Wikipedia results found for your query."
        title   = docs[0].metadata.get("title", "Wikipedia")
        summary = docs[0].page_content
        return title, summary
    except Exception as ex:
        return None, f"Wikipedia search error: {ex}"


# ── Model calls ───────────────────────────────────────────────────────────────

def ask_model_direct(user_query: str, model: str = MODEL_NAME) -> str:
    """
    Ask the local model to answer from its own knowledge.
    Returns NEED_WIKI if it's not confident.

    Args:
        user_query: The user's question.
        model: Ollama model name to use (defaults to MODULE_NAME).
    """
    system_prompt = (
        "You are a knowledgeable assistant. "
        "Answer the user's question clearly and concisely using your own knowledge. "
        "However, if you are NOT confident in your answer, or the topic is too specific, "
        "niche, or recent for you to answer accurately, respond with ONLY the word: NEED_WIKI "
        "(nothing else, no explanation). Do NOT use NEED_WIKI if you genuinely know the answer."
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ]
    )
    return response["message"]["content"]


def ask_model_with_context(user_query: str, wiki_title: str, wiki_context: str,
                           model: str = MODEL_NAME) -> str:
    """
    Ask the local model to answer using Wikipedia context.

    Args:
        user_query: The user's question.
        wiki_title: The Wikipedia article title.
        wiki_context: The Wikipedia article summary text.
        model: Ollama model name to use (defaults to MODEL_NAME).
    """
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


# ── Chatbot loop (console) ────────────────────────────────────────────────────

def start_chatbot():
    print(f"🤖 Local Chatbot (powered by {MODEL_NAME} + Wikipedia fallback)")
    print("   No API key needed — runs entirely on your device.")
    print("   I'll answer from my own knowledge first.")
    print("   If I'm unsure, I'll search Wikipedia and answer from there.")
    print("   Type 'exit' or 'quit' to stop.\n")
    print("-" * 60)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            # ── Step 1: Ask the model directly ──────────────────────────────
            print(f"\n🧠 Asking {MODEL_NAME} from its own knowledge...")
            reply = ask_model_direct(user_input)

            if reply.strip() == NEED_WIKI:
                # ── Step 2: Model unsure → fetch Wikipedia ───────────────────
                cleaned = clean_query(user_input)
                print(f"🔍 Model unsure — searching Wikipedia for: '{cleaned}'...")
                wiki_title, wiki_context = search_wikipedia(user_input)

                if wiki_title is None:
                    print(f"⚠️  {wiki_context}")
                    print(f"{MODEL_NAME}: Sorry, I don't have enough information to answer that.")
                    print("📌 Source: Neither model nor Wikipedia could answer")
                else:
                    print(f"✅ Found Wikipedia article: '{wiki_title}'")
                    print("💬 Generating answer from Wikipedia context...\n")
                    reply = ask_model_with_context(user_input, wiki_title, wiki_context)
                    print(f"{MODEL_NAME}: {reply}")
                    print(f"📌 Source: Answered using Wikipedia — '{wiki_title}'")

            else:
                # ── Model answered confidently ────────────────────────────────
                print(f"{MODEL_NAME}: {reply}")
                print("📌 Source: Answered from model's own knowledge")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("   Make sure Ollama is running: `ollama serve`")

        print("-" * 60)


if __name__ == "__main__":
    start_chatbot()
