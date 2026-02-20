# GenAI Practice

A collection of AI chatbot experiments using local Ollama models and the OpenCode API.

## Projects

| File | Description |
|---|---|
| `opencode_chatbot.py` | CLI chatbot using Big Pickle model via OpenCode API |
| `wiki_chatbot.py` | CLI chatbot — model answers first, falls back to Wikipedia |
| `local_wiki_chatbot.py` | Same as above but uses local Ollama model (no API key needed) |
| `local_wiki_streamlit.py` | Streamlit UI for the local wiki chatbot |

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd GenAI-practice
```

### 2. Create a virtual environment
```bash
python -m venv genai-clean-env
.\genai-clean-env\Scripts\Activate.ps1   # Windows
source genai-clean-env/bin/activate       # Mac/Linux
```

### 3. Install dependencies
```bash
pip install openai wikipedia python-dotenv streamlit ollama
```

### 4. Set up your API key
```bash
copy .env.example .env    # Windows
cp .env.example .env      # Mac/Linux
# Then edit .env and add your actual OPENCODE_API_KEY
```

### 5. Install Ollama (for local models)
- Download from https://ollama.com
- Pull a model: `ollama pull qwen2.5:3b`

## Running

```bash
# CLI chatbot (OpenCode API)
python opencode_chatbot.py

# CLI chatbot (local Ollama model + Wikipedia)
python local_wiki_chatbot.py

# Streamlit UI (local Ollama model + Wikipedia)
streamlit run local_wiki_streamlit.py
```
