# langchain-streamlit-rag

A Streamlit app demonstrating Retrieval-Augmented Generation (RAG) for answering questions about Klaipėda, Lithuania. The app uses LangChain to load and embed documents from multiple sources (web, PDF, and text), stores them in an in-memory vector store, and provides a conversational interface for question answering.

## Features
- Loads knowledge from:
  - Wikipedia and Klaipėda travel web pages
  - A local PDF file: `Pesciomis-po-Klaipeda-LT.pdf`
- Splits documents into chunks and embeds them using OpenAI embeddings
- Stores embeddings in an in-memory vector store
- Uses a chat model (via GitHub API) to answer questions based on retrieved context
- Displays sources for each answer

## Setup

### 1. Clone the repository
```sh
git clone <repo-url>
cd langchain-streamlit-rag
```

### 2. Install Python (recommended: 3.13)
Ensure you have Python 3.13 installed. You can use [pyenv](https://github.com/pyenv/pyenv) or check `.python-version`.

### 3. Install dependencies
We recommend using [uv](https://github.com/astral-sh/uv) for fast installs:
```sh
uv sync
```
Or use pip:
```sh
pip install -r pyproject.toml
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and fill in your API keys:
```sh
cp .env.example .env
```

### 5. Run the app
```sh
streamlit run main.py
```

## Environment Variables
The app requires the following environment variables (see `.env.example`):
- `OPENAI_API_KEY` – Your OpenAI API key for embeddings
- `GITHUB_TOKEN` – Your GitHub token for using the GitHub-hosted chat model

## Data Sources
- [Wikipedia: Klaipėda](https://lt.wikipedia.org/wiki/Klaip%C4%97da)
- [Klaipėda Travel](https://klaipedatravel.lt/)
- `Pesciomis-po-Klaipeda-LT.pdf` (local PDF, source https://klaipedatravel.lt/)
