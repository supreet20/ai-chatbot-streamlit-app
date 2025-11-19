# RAG Chatbot

A Retrieval-Augmented Generation (RAG) assistant that lets you ask questions about your own PDF/TXT documents. The project ships with both a command-line chatbot (`app.py`) and an interactive Streamlit UI (`streamlit_app.py`).

## Features

- **Document ingestion** – Drop PDFs or text files inside `rag-chatbot/data/` and the app automatically chunks, embeds, and stores them in ChromaDB using MiniLM embeddings.
- **Metadata-aware retrieval** – Each response shows which file/page the information came from for better grounding.
- **File-aware intent handling** – Requests like “What files do we have?” or “Summarise each document” are handled without prompting the LLM, while file-specific questions (e.g., “From the iPhone 16 Pro file…”) automatically scope retrieval to that source.
- **Streamlit uploader** – Drag-and-drop files directly in the UI; the vector store rebuilds on demand when documents change.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed locally with the `llama3.2` (or compatible) model pulled.
- Recommended: create a virtual environment for dependencies.

Install required Python packages (from inside `rag-chatbot/`):

```bash
pip install -r requirements.txt
```

(If a `requirements.txt` does not exist yet, install the packages listed at the top of `app.py`/`streamlit_app.py`.)

## Directory Layout

```
rag-chatbot/
├── app.py                # CLI chatbot
├── streamlit_app.py      # Streamlit UI
├── data/                 # Place your PDFs/TXTs here
├── chroma_db/            # Persisted vector store for CLI
├── chroma_db_streamlit/  # Persisted vector store for Streamlit
└── README.md
```

## Running the CLI Chatbot

```bash
python app.py
```

The CLI will build the vector store the first time documents are detected, then drop you into an interactive prompt. Type `exit` to quit.

## Running the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Key UI elements:

- **Sidebar** – Upload new files, view the current catalog, or force a vector-store rebuild.
- **Chat box** – Ask questions across all documents, or include a file’s name in your prompt to scope to that file (e.g., “From `iPhone 16 Pro - Tech Specs`…”).
- **Special phrases** – “Summarise each file” summarizes every document; “What files do we have?” lists the current catalog.

## Adding Documents

1. Copy PDFs or UTF-8 text files into `rag-chatbot/data/`, or use the Streamlit uploader.
2. (Optional) Delete the `chroma_db*/` directories if you want to force a fresh embedding pass.
3. Restart the CLI/UI; it detects new files via a manifest and rebuilds automatically when necessary.

## Troubleshooting

- **“I don’t know the context” responses** – Ensure documents exist in `data/` and rerun the “Rebuild vector store” button or delete the `chroma_db*` folders before restarting.
- **Ollama connection errors** – Start the Ollama server (`ollama serve`) and make sure the `llama3.2` model is available locally.
- **Large PDFs** – Adjust `chunk_size`/`chunk_overlap` inside `app.py`/`streamlit_app.py` if your documents need different splitting.

## Future Ideas

- Authentication for the Streamlit UI
- Exportable chat transcripts with cited sources
- Support for additional document types (Markdown, DOCX, etc.)

Feel free to fork and extend the chatbot to suit your workflows!***
