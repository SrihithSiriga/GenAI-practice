import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Access the API key you added
api_key = os.getenv("OPENCODE_API_KEY")

if not api_key:
    print("Error: OPENCODE_API_KEY not found. Make sure it's in your .env file.")
    exit(1)

# Initialize the OpenAI client targeting OpenCode's endpoint
# Note: You may need to verify the exact base_url for OpenCode Zen if it differs
client = OpenAI(
    api_key=api_key,
    base_url="https://opencode.ai/zen/v1" # Correct OpenCode Zen API endpoint
)

def start_chatbot():
    print("🤖 Chatbot initialized with Big Pickle model. Type 'exit' to quit.")
    print("-" * 50)

    total_session_tokens = 0  # Running cumulative token count for the session

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in ['exit', 'quit']:
            print(f"\n📊 Session Total Tokens Used: {total_session_tokens}")
            print("Goodbye!")
            break

        try:
            # Use with_raw_response to also capture HTTP headers (for tokens remaining)
            raw = client.chat.completions.with_raw_response.create(
                model="big-pickle",
                messages=[
                    {"role": "system", "content": "You are a helpful chatbot."},
                    {"role": "user", "content": user_input}
                ]
            )
            response = raw.parse()

            # Print the model's response
            model_reply = response.choices[0].message.content
            print(f"Big Pickle: {model_reply}")

            # Token usage for this call
            usage = response.usage
            total_session_tokens += usage.total_tokens

            # Try to read remaining tokens from response headers
            tokens_remaining = raw.headers.get("x-ratelimit-remaining-tokens", "N/A")
            tokens_limit     = raw.headers.get("x-ratelimit-limit-tokens", "N/A")

            print(f"\n📊 Token Usage — Prompt: {usage.prompt_tokens} | Completion: {usage.completion_tokens} | Total this call: {usage.total_tokens} | Session total: {total_session_tokens}")
            print(f"🔋 Tokens Remaining: {tokens_remaining} / {tokens_limit}")
            print("-" * 50)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    start_chatbot()
