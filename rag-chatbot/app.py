import os
from typing import Dict, List

# LangChain core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Community integrations
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from langchain_text_splitters import RecursiveCharacterTextSplitter



# ---------- 1. Load documents ----------

def load_docs(data_dir: str) -> List[Document]:
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif fname.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        # you can extend for .md, .docx etc.
    return docs


# ---------- 2. Split into chunks ----------

def split_docs(docs: List[Document], chunk_size=800, chunk_overlap=150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)


# ---------- 3. Create / load vector store ----------

def build_vector_store(chunks: List[Document], persist_dir: str = "chroma_db") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    return vectorstore


def load_vector_store(persist_dir: str = "chroma_db") -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


# ---------- 4. RAG chain: retrieve + answer ----------

def format_chunk_with_metadata(doc: Document) -> str:
    """Include file metadata so the LLM knows the source of each chunk."""
    source = doc.metadata.get("source") or "Unknown source"
    source = os.path.basename(source)
    page = doc.metadata.get("page")
    header = f"Source: {source}"
    if page is not None:
        header += f" (page {page})"
    return f"{header}\n{doc.page_content}"


def create_rag_chain(vectorstore: Chroma, llm: ChatOllama):
    # New-style retriever (Runnable)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

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

    def answer(question: str) -> str:
        # 1. Retrieve relevant chunks (LangChain 0.3+ style)
        docs = retriever.invoke(question)

        if not docs:
            return "I couldn't find anything relevant in the documents for that question."

        # 2. Build context from docs
        context_text = "\n\n".join(format_chunk_with_metadata(d) for d in docs)

        # 3. Format prompt to ChatPromptValue for LLM call
        formatted = prompt.invoke({"context": context_text, "question": question})

        # 4. Call LLM
        response = llm.invoke(formatted)
        return response.content

    return answer


def wants_file_summaries(question: str) -> bool:
    lowered = question.lower()
    return "summary" in lowered and "file" in lowered


def summarize_files(docs: List[Document], llm: ChatOllama, max_chars_per_file: int = 3000) -> str:
    if not docs:
        return "I don't have any documents to summarize."

    grouped: Dict[str, List[str]] = {}
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source") or "Unknown source")
        grouped.setdefault(source, []).append(doc.page_content)

    summary_prompt = ChatPromptTemplate.from_messages(
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
        formatted = summary_prompt.invoke({"file_name": file_name, "content": excerpt})
        result = llm.invoke(formatted)
        outputs.append(f"{file_name}:\n{result.content.strip()}")

    return "\n\n".join(outputs)


# ---------- 5. CLI app ----------

def main():
    data_dir = "data"
    persist_dir = "chroma_db"
    docs = load_docs(data_dir)

    # If we don't have a persisted DB yet, build it
    if not os.path.exists(persist_dir) or len(os.listdir(persist_dir)) == 0:
        print("Building vector store from documents...")
        if not docs:
            print(f"No documents found in {data_dir}. Add PDFs or TXT files and try again.")
            return
        chunks = split_docs(docs)
        vectorstore = build_vector_store(chunks, persist_dir=persist_dir)
        print("Vector store built and persisted.")
    else:
        print("Loading existing vector store...")
        vectorstore = load_vector_store(persist_dir=persist_dir)
        print("Vector store loaded.")

    llm = ChatOllama(
        model="llama3.2",
        temperature=0.1,
    )
    rag_answer = create_rag_chain(vectorstore, llm)

    print("\nRAG chatbot ready. Type your questions (or 'exit' to quit).\n")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if q.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not q.strip():
            continue

        print("Thinking...")
        if wants_file_summaries(q):
            answer = summarize_files(docs, llm)
        else:
            answer = rag_answer(q)
        print(f"\nBot: {answer}\n")


if __name__ == "__main__":
    main()
