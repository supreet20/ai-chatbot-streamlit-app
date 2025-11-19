import json
import os
from pathlib import Path
from typing import Dict, List

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db_streamlit"  # keep separate from CLI if you like
MANIFEST_PATH = PERSIST_DIR / "manifest.json"


# ---------- 1. Helpers to load & prepare docs ----------

def iter_data_files(data_dir: Path = DATA_DIR):
    if not data_dir.exists():
        return []
    supported = {".pdf", ".txt"}
    return [path for path in data_dir.iterdir() if path.suffix.lower() in supported]


def load_docs(data_dir: Path = DATA_DIR) -> List[Document]:
    docs = []
    for path in iter_data_files(data_dir):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
    return docs


def format_file_list(file_names: List[str]) -> str:
    if not file_names:
        return "No files available. Upload PDF or TXT files to the data folder."
    lines = "\n".join(f"- {name}" for name in file_names)
    return f"Available files:\n{lines}"


def file_name_map(file_names: List[str]) -> Dict[str, str]:
    mapping = {}
    for name in file_names:
        normalized = "".join(ch.lower() for ch in Path(name).stem if ch.isalnum())
        if normalized:
            mapping[normalized] = name
    return mapping


def extract_requested_files(question: str, file_names: List[str]) -> List[str]:
    normalized_question = "".join(ch.lower() for ch in question if ch.isalnum())
    mapping = file_name_map(file_names)
    matched = [original for norm, original in mapping.items() if norm and norm in normalized_question]
    return matched


def split_docs(docs: List[Document], chunk_size=800, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks: List[Document], persist_dir: Path = PERSIST_DIR) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    vectordb.persist()
    return vectordb


def load_vector_store(persist_dir: Path = PERSIST_DIR) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


# ---------- 2. RAG chain factory ----------

@st.cache_resource
def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.1,
    )

def format_chunk_with_metadata(doc: Document) -> str:
    source = Path(doc.metadata.get("source", "unknown")).name
    page = doc.metadata.get("page")
    header = f"Source: {source}"
    if page is not None:
        header += f" (page {page})"
    return f"{header}\n{doc.page_content}"


def create_rag_chain(vectorstore: Chroma, llm: ChatOllama, available_sources: List[str]):

    system_prompt = """
You are an AI assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say you don't know and suggest what the user could look for.
Be concise and clear.

Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    def retrieve(question: str, scoped_sources: List[str] = None, per_file_k: int = 3):
        docs = []
        targets = scoped_sources if scoped_sources else available_sources
        if targets:
            for source_path in targets:
                docs.extend(
                    vectorstore.similarity_search(
                        question,
                        k=per_file_k,
                        filter={"source": source_path},
                    )
                )
        if not docs:
            docs = vectorstore.similarity_search(question, k=per_file_k * 2)
        return docs

    def answer(question: str, scoped_sources: List[str] = None) -> str:
        docs = retrieve(question, scoped_sources=scoped_sources)

        if not docs:
            return "I couldn't find anything relevant in the documents for that question."

        context_text = "\n\n".join(format_chunk_with_metadata(d) for d in docs)
        available_files_text = st.session_state.get("available_files_text")
        if available_files_text:
            context_text = f"{available_files_text}\n\n{context_text}"
        formatted = prompt.invoke({"context": context_text, "question": question})
        response = llm.invoke(formatted)
        return response.content

    return answer


# ---------- 3. Cached resources for Streamlit ----------

# ---------- 3. Cached resources for Streamlit ----------

def has_persisted_db(persist_dir: Path) -> bool:
    return persist_dir.exists() and any(persist_dir.iterdir())


def current_manifest(data_dir: Path = DATA_DIR) -> Dict[str, Dict[str, float]]:
    manifest = {}
    for path in iter_data_files(data_dir):
        stats = path.stat()
        manifest[path.name] = {"size": stats.st_size, "mtime": stats.st_mtime}
    return manifest


def read_manifest(path: Path = MANIFEST_PATH) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def manifests_match(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]) -> bool:
    return a == b


def write_manifest(data: Dict[str, Dict[str, float]], path: Path = MANIFEST_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@st.cache_resource(show_spinner="Loading vector store...")
def get_vectorstore():
    current = current_manifest(DATA_DIR)
    stored = read_manifest()

    if has_persisted_db(PERSIST_DIR) and manifests_match(current, stored):
        return load_vector_store(PERSIST_DIR)

    docs = load_docs(DATA_DIR)
    if not docs:
        return None

    chunks = split_docs(docs)
    vectordb = build_vector_store(chunks, persist_dir=PERSIST_DIR)
    write_manifest(current)
    return vectordb


def wants_file_summaries(text: str) -> bool:
    lowered = text.lower()
    return "summary" in lowered and ("file" in lowered or "document" in lowered)


def wants_file_list(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in [
            "what files",
            "which files",
            "list files",
            "what documents",
            "which documents",
            "list documents",
        ]
    )


def summarize_files(docs: List[Document], llm: ChatOllama, max_chars_per_file: int = 6000) -> str:
    if not docs:
        return "I don't have any documents to summarize. Please upload files first."

    grouped = {}
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source") or "unknown")
        grouped.setdefault(source, []).append(doc.page_content)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You create concise summaries (2-3 sentences) for the provided file content. "
                "Use only the supplied text.",
            ),
            (
                "human",
                "File: {file_name}\n\nContent excerpts:\n{content}\n\nSummary:",
            ),
        ]
    )

    outputs = []
    for file_name, contents in grouped.items():
        joined = "\n\n".join(contents)
        excerpt = joined[:max_chars_per_file]
        formatted = prompt.invoke({"file_name": file_name, "content": excerpt})
        result = llm.invoke(formatted)
        outputs.append(f"{file_name}:\n{result.content.strip()}")

    return "\n\n".join(outputs)


# ---------- 4. Streamlit UI ----------


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

    st.title("📄 RAG Chatbot over Your Documents")
    st.write("Ask questions or request summaries about the files in your `data/` folder.")
    docs = load_docs(DATA_DIR)
    file_names = [path.name for path in iter_data_files(DATA_DIR)]
    source_paths = [str((DATA_DIR / name).resolve()) for name in file_names if (DATA_DIR / name).exists()]
    st.session_state["available_files_text"] = format_file_list(file_names)

    with st.sidebar:
        st.header("📁 Data status")

        uploaded_files = st.file_uploader(
            "Upload PDF/TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Files are stored in the data folder used for retrieval.",
        )
        if uploaded_files:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            saved = []
            for uploaded in uploaded_files:
                file_path = DATA_DIR / uploaded.name
                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                saved.append(uploaded.name)

            if saved:
                st.success(f"Uploaded: {', '.join(saved)}")
                st.cache_resource.clear()
                docs = load_docs(DATA_DIR)
                file_names = [path.name for path in iter_data_files(DATA_DIR)]
                source_paths = [str((DATA_DIR / name).resolve()) for name in file_names if (DATA_DIR / name).exists()]
                st.session_state["available_files_text"] = format_file_list(file_names)
            else:
                st.info("No files uploaded.")

        st.write(f"Data folder: `{DATA_DIR}`")
        file_count = len(file_names)
        st.write(f"Number of documents found: **{file_count}**")

        if file_names:
            st.write("Files:")
            for s in file_names:
                st.markdown(f"- `{s}`")
        else:
            st.warning("No documents found. Add PDFs or TXT files to the `data/` folder and refresh.")

        if st.button("Rebuild vector store"):
            # clear cache and rebuild
            st.cache_resource.clear()
            st.success("Cache cleared. The vector store will rebuild on next question.")
            docs = load_docs(DATA_DIR)
            file_names = [path.name for path in iter_data_files(DATA_DIR)]
            source_paths = [str((DATA_DIR / name).resolve()) for name in file_names if (DATA_DIR / name).exists()]
            st.session_state["available_files_text"] = format_file_list(file_names)

    vectorstore = get_vectorstore()

    if vectorstore is None:
        st.error("No documents available. Please add PDFs/TXT to the `data/` folder and refresh.")
        return
    llm = get_llm()
    rag_answer = create_rag_chain(vectorstore, llm, source_paths)

    # Chat-style interface
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask a question about your documents...", key="chat_input")
    if user_input:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                requested_files = extract_requested_files(user_input, file_names)

                if wants_file_list(user_input):
                    answer = format_file_list(file_names)
                elif wants_file_summaries(user_input):
                    answer = summarize_files(docs, llm)
                else:
                    scoped_sources = [
                        str((DATA_DIR / fname).resolve())
                        for fname in requested_files
                        if (DATA_DIR / fname).exists()
                    ]
                    answer = rag_answer(user_input, scoped_sources=scoped_sources if scoped_sources else None)
                st.markdown(answer)

        st.session_state["messages"].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
