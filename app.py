#!/usr/bin/env python3
"""
News RAG CLI: Terminal-based conversational RAG over NewsAPI articles using Chroma.
Supports managing multiple search queries and multi-turn Q&A sessions.
Persisted queries across restarts via a JSON manifest.

Requirements:
- langchain
- langchain-openai
- langchain-core
- langchain-chroma
- requests
- beautifulsoup4
- python-dotenv

Usage:
A technical CLI tool for experimenting with Retrieval-Augmented Generation over recent news articles.
"""

import os
import re
import shutil
import requests
import sys
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Model configuration from environment
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# Constants
NEWSAPI_ENDPOINT_HEADLINES = "https://newsapi.org/v2/top-headlines"
NEWSAPI_ENDPOINT_EVERYTHING = "https://newsapi.org/v2/everything"
MAX_QUERIES = 10
MAX_ARTICLES = 100
MANIFEST_PATH = "./queries_manifest.json"
# Base directory for all Chroma stores
BASE_STORE_DIR = "./stores"

# Ensure base store directory exists
os.makedirs(BASE_STORE_DIR, exist_ok=True)


def clear_console():
    """Clear the terminal screen based on OS type."""
    os.system('cls' if os.name == 'nt' else 'clear')


class ArticleQuery:
    """
    Encapsulates a news search query and its Chroma index.
    Responsible for fetching articles, processing text, and persisting vector embeddings.
    """

    def __init__(self, query: str, api_key: str):
        """Initialize with a query string and NewsAPI key."""
        self.query = query
        self.api_key = api_key
        safe = re.sub(r"\W+", "_", query.lower())
        self.collection_name = f"chroma_store_{safe}"
        self.store_dir = os.path.join(BASE_STORE_DIR, self.collection_name)
        self.vectorstore = None

    def _make_directory_name(self, query: str) -> str:
        """Convert the search query into a safe subdirectory under BASE_STORE_DIR."""
        safe = re.sub(r"\W+", "_", query.lower())
        return os.path.join(BASE_STORE_DIR, f"chroma_store_{safe}")

    def fetch_articles(self) -> list[dict]:
        """Retrieve up to MAX_ARTICLES using NewsAPI's headlines and everything endpoints."""
        headers = {"X-Api-Key": self.api_key}
        params = {"q": self.query, "pageSize": MAX_ARTICLES, "language": "en"}
        r1 = requests.get(NEWSAPI_ENDPOINT_HEADLINES, headers=headers, params=params)
        r1.raise_for_status()
        articles = r1.json().get("articles", [])

        if len(articles) < MAX_ARTICLES:
            remaining = MAX_ARTICLES - len(articles)
            params.update({"pageSize": remaining})
            r2 = requests.get(NEWSAPI_ENDPOINT_EVERYTHING, headers=headers, params=params)
            r2.raise_for_status()
            articles.extend(r2.json().get("articles", []))
        return articles

    def fetch_full_text(self, url: str) -> str:
        """Extract the full text from a given article URL using BeautifulSoup."""
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            return soup.get_text(separator="\n")
        except requests.RequestException:
            return ""

    def build_vectorstore(self):
        """Fetch articles, split them into chunks, and persist embeddings in a Chroma vector store."""
        articles = self.fetch_articles()
        chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for art in articles:
            content = art.get("content") or art.get("description") or ""
            text = self.fetch_full_text(art.get("url", "")) or content
            for seg in splitter.split_text(text):
                chunks.append(Document(page_content=seg, metadata={"source": art.get("url")}))

        # Recreate store directory for this query
        if os.path.isdir(self.store_dir):
            shutil.rmtree(self.store_dir)
        os.makedirs(self.store_dir, exist_ok=True)

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.store_dir,
            collection_name=self.collection_name
        )

    def load_vectorstore(self):
        """Load an existing vector store if it exists on disk."""
        if os.path.isdir(self.store_dir):
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=self.store_dir,
                collection_name=self.collection_name
            )

    def delete_store(self):
        """Delete the persistent Chroma vector store directory for this query."""
        if os.path.isdir(self.store_dir):
            shutil.rmtree(self.store_dir)


def load_manifest() -> list[dict]:
    """Read and return saved search queries from the JSON manifest file."""
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return []


def save_manifest(manifest: list[dict]):
    """Save the current search queries into the JSON manifest file."""
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)


class NewsRAGApp:
    """
    Terminal-based manager for multi-query conversational RAG.
    Manages querying, embedding, chat sessions, and persistent storage.
    """

    def __init__(self):
        """Initialize the app by loading saved queries and setting up the OpenAI client."""
        key = os.getenv("NEWSAPI_KEY")
        if not key:
            print("Error: Set NEWSAPI_KEY environment variable.")
            sys.exit(1)
        self.api_key = key
        self.queries: dict[int, ArticleQuery] = {}
        self.selected_id: int | None = None
        self.llm = ChatOpenAI(model=LLM_MODEL)
        for entry in load_manifest():
            qid, qtext = entry['id'], entry['query']
            aq = ArticleQuery(qtext, self.api_key)
            aq.load_vectorstore()
            self.queries[qid] = aq

    def menu(self):
        """Display the main menu interface and handle user commands interactively."""
        while True:
            print("\n=== News RAG CLI ===")
            for qid, q in self.queries.items():
                sel = " (selected)" if qid == self.selected_id else ""
                print(f"{qid}. {q.query}{sel}")
            print("\na) Add   d<ID> Delete   s<ID> Select   r Run   q Quit")

            choice = input("Choice: ").strip()
            if choice == 'a':
                self.add_query()
            elif choice.startswith('d'):
                self.delete_query(choice)
            elif choice.startswith('s'):
                self.select_query(choice)
            elif choice == 'r':
                self.run_session()
            elif choice == 'q':
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")

    def add_query(self):
        """Prompt the user to add a new search query and build its index."""
        if len(self.queries) >= MAX_QUERIES:
            print(f"Max {MAX_QUERIES} reached.")
            return
        q = input("New query: ").strip()
        if not q:
            return
        qid = max(self.queries.keys(), default=0) + 1
        print("Indexing articles...")
        aq = ArticleQuery(q, self.api_key)
        aq.build_vectorstore()
        self.queries[qid] = aq
        save_manifest([{'id': k, 'query': v.query} for k, v in self.queries.items()])
        print(f"Added [{qid}] {q}.")

    def delete_query(self, cmd: str):
        """Delete an existing query and remove its stored vector index."""
        try:
            qid = int(cmd[1:])
            if qid in self.queries:
                self.queries[qid].delete_store()
                del self.queries[qid]
                if self.selected_id == qid:
                    self.selected_id = None
                save_manifest([{'id': k, 'query': v.query} for k, v in self.queries.items()])
                print(f"Deleted [{qid}].")
            else:
                print("No such ID.")
        except ValueError:
            print("Format: d<ID>")

    def select_query(self, cmd: str):
        """Mark a query as currently selected for the chat session."""
        try:
            qid = int(cmd[1:])
            if qid in self.queries:
                self.selected_id = qid
                print(f"Selected [{qid}].")
            else:
                print("No such ID.")
        except ValueError:
            print("Format: s<ID>")

    def run_session(self):
        """Start a conversational Q&A session based on the selected query."""
        if self.selected_id is None:
            print("Select a query first.")
            return
        aq = self.queries[self.selected_id]
        retriever = aq.vectorstore.as_retriever()
        conv = ConversationalRetrievalChain.from_llm(llm=self.llm, retriever=retriever)
        history = []
        print(f"\n=== Conversation for '{aq.query}' === (type 'exit' to go back)")
        while True:
            prompt = input("You: ").strip()
            if prompt.lower() in ('exit', 'quit'):
                break
            docs = retriever.invoke(prompt)
            if not docs:
                print("\nAI: No information found.")
                print("\n-----------\n")
                continue
            res = conv.invoke({"question": prompt, "chat_history": history})
            history = res.get("chat_history", history)
            print(f"\nAI: {res.get('answer')}\n\n-----------\n")


if __name__ == "__main__":
    NewsRAGApp().menu()
