import os
import wikipedia
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("OPENCODE_API_KEY")

if not api_key:
    print("Error: OPENCODE_API_KEY not found. Make sure it's in your .env file.")
    exit(1)

# Initialize the OpenAI client targeting OpenCode's endpoint
client = OpenAI(
    api_key=api_key,
    base_url="https://opencode.ai/zen/v1"
)


def clean_query(query: str) -> str:
    """
    Strip common conversational filler phrases so Wikipedia gets a clean topic.
    e.g. 'Tell me about element Mercury' -> 'element Mercury'
    """
    import re
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
    return cleaned or query  # fallback to original if cleaning empties it


def search_wikipedia(query: str, sentences: int = 10) -> str:
    """
    Search Wikipedia for the given query and return a summary.
    Falls back gracefully on disambiguation or page errors.
    """
    try:
        # Clean conversational filler before searching
        search_term = clean_query(query)
        results = wikipedia.search(search_term, results=3)
        if not results:
            return None, "No Wikipedia results found for your query."

        # Try to get the page summary for the top result
        page = wikipedia.page(results[0], auto_suggest=False)
        summary = wikipedia.summary(results[0], sentences=sentences, auto_suggest=False)
        return page.title, summary

    except wikipedia.exceptions.DisambiguationError as e:
        # If the term is ambiguous, pick the first option
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
            return page.title, summary
        except Exception:
            return None, f"Topic is ambiguous. Options include: {', '.join(e.options[:5])}"

    except wikipedia.exceptions.PageError:
        return None, f"No Wikipedia page found for '{query}'."

    except Exception as ex:
        return None, f"Wikipedia search error: {ex}"


NEED_WIKI = "NEED_WIKI"


def ask_model_direct(user_query: str) -> tuple:
    """
    Ask the model to answer from its own knowledge.
    If it is not confident, it must reply with exactly 'NEED_WIKI'.
    Returns (reply_text, usage, headers).
    """
    system_prompt = (
        "You are a knowledgeable assistant. "
        "Answer the user's question clearly and concisely using your own knowledge. "
        "However, if you are NOT confident in your answer or the topic is too specific, "
        "niche, or recent for you to answer accurately, respond with ONLY the word: NEED_WIKI "
        "(nothing else, no explanation). Do NOT use NEED_WIKI if you genuinely know the answer."
    )

    raw = client.chat.completions.with_raw_response.create(
        model="big-pickle",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query}
        ]
    )
    response = raw.parse()
    return response.choices[0].message.content, response.usage, raw.headers


def ask_model_with_context(user_query: str, wiki_title: str, wiki_context: str) -> tuple:
    """
    Send the user query + Wikipedia context to the Big Pickle model.
    Returns (reply_text, usage, headers).
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

    raw = client.chat.completions.with_raw_response.create(
        model="big-pickle",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )
    response = raw.parse()
    return response.choices[0].message.content, response.usage, raw.headers


def start_chatbot():
    print("🤖 Smart Chatbot (powered by Big Pickle + Wikipedia fallback)")
    print("   I'll answer from my own knowledge first.")
    print("   If I'm unsure, I'll search Wikipedia and answer from there.")
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

        try:
            # ── Step 1: Ask the model directly ──────────────────────────────
            print("\n🧠 Asking Big Pickle from its own knowledge...")
            reply, usage, headers = ask_model_direct(user_input)
            total_session_tokens += usage.total_tokens

            if reply.strip() == NEED_WIKI:
                # ── Step 2: Model is unsure → fetch Wikipedia ─────────────
                cleaned = clean_query(user_input)
                print(f"🔍 Model unsure — searching Wikipedia for: '{cleaned}'...")
                wiki_title, wiki_context = search_wikipedia(user_input)

                if wiki_title is None:
                    print(f"⚠️  {wiki_context}")
                    print("Big Pickle: Sorry, I don't have enough information to answer that.")
                else:
                    print(f"✅ Found Wikipedia article: '{wiki_title}'")
                    print("💬 Generating answer from Wikipedia context...\n")
                    reply, usage2, headers = ask_model_with_context(user_input, wiki_title, wiki_context)
                    total_session_tokens += usage2.total_tokens
                    usage = usage2  # use final call's token counts for display
                    print(f"Big Pickle: {reply}")
                    print(f"📌 Source: Answered using Wikipedia — '{wiki_title}'")
            else:
                # ── Model answered confidently ────────────────────────────
                print(f"Big Pickle: {reply}")
                print("📌 Source: Answered from model's own knowledge")

            tokens_remaining = headers.get("x-ratelimit-remaining-tokens", "N/A")
            tokens_limit     = headers.get("x-ratelimit-limit-tokens", "N/A")

            print(f"\n📊 Tokens — Prompt: {usage.prompt_tokens} | Completion: {usage.completion_tokens} | Total this call: {usage.total_tokens} | Session total: {total_session_tokens}")
            print(f"🔋 Tokens Remaining: {tokens_remaining} / {tokens_limit}")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 60)


if __name__ == "__main__":
    start_chatbot()
