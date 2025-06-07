# News RAG CLI

**News RAG CLI** is a terminal-based application that uses Retrieval-Augmented Generation (RAG) to let you query real news articles using natural language. It fetches headlines and full articles via NewsAPI, embeds them using OpenAI embeddings, stores them in Chroma vector DBs, and provides a multi-turn conversational interface using OpenAI's LLMs.

---

## 📦 Features

- 🔍 Fetch news from NewsAPI for up to 10 queries.
- 🧠 Embeds and stores article chunks using OpenAI and Chroma.
- 💬 Multi-turn conversations with memory using LangChain.
- 💾 Query data persists across restarts via JSON manifest and Chroma stores.
- 🧪 Clean CLI experience with ID-based query selection.

---

## 🚀 Setup

### 1. Clone the Repo

```bash
git clone https://github.com/tahierhussain2205/news-rag-cli.git
cd news-rag-cli
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API keys

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
NEWSAPI_KEY=your-newsapi-key
```

---

## 🖥️ Usage

Run the app:

```bash
python app.py
```

You'll see a terminal UI like:

```
=== News RAG Manager ===
1. climate change
2. artificial intelligence (selected)

a) Add   d<ID> Delete   s<ID> Select   r Run   q Quit
Choice:
```

### Commands

- `a` — Add a new query and index articles
- `d<ID>` — Delete query (e.g. `d2`)
- `s<ID>` — Select query (e.g. `s1`)
- `r` — Start a conversation with the selected query's articles
- `q` — Quit

---

## 💡 Example: Step-by-Step Usage

Here’s what a typical session looks like in the terminal:

```bash
$ python app.py

=== News RAG CLI ===

a) Add   d<ID> Delete   s<ID> Select   r Run   q Quit
Choice: a

New query: space exploration
Indexing articles...
Added [1] space exploration.

=== News RAG CLI ===
1. space exploration

a) Add   d<ID> Delete   s<ID> Select   r Run   q Quit
Choice: s1
Selected [1].

=== News RAG CLI ===
1. space exploration (selected)

a) Add   d<ID> Delete   s<ID> Select   r Run   q Quit
Choice: r

=== Conversation for 'space exploration' ===
(type 'exit' to go back)

You: What recent missions has NASA launched?

AI: NASA recently launched the Artemis I mission to test...

-----------

You: What are the key challenges in Mars exploration?

AI: Some major challenges in Mars exploration include...

-----------

You: exit

=== News RAG CLI ===
1. space exploration

a) Add   d<ID> Delete   s<ID> Select   r Run   q Quit
Choice: q

Goodbye!

```

---

## 🗃️ File Structure

```
├── app.py       # Main application
├── requirements.txt      # Python dependencies
├── sample.env            # Sample .env file
├── .gitignore
├── queries_manifest.json # Query metadata (auto-generated)
├── stores/               # Chroma vector databases
```

---

## 📜 License

MIT License © 2025 Tahier Hussain

---
