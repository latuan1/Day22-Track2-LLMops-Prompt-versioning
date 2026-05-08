"""Step 1: LangSmith-instrumented RAG pipeline."""

from __future__ import annotations

from config import (
    configure_langsmith_tracing,
    ensure_directories,
    get_settings,
    limit_items,
    load_knowledge_base,
    make_embeddings,
    make_llm,
)

settings = get_settings(require_openai=True, require_langsmith=True)
configure_langsmith_tracing(settings)
ensure_directories()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer using only the context below. "
            "If the answer is not in the context, say you do not have enough "
            "information.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def build_vectorstore() -> FAISS:
    """Load, split, embed, and index the knowledge base with FAISS."""

    text = load_knowledge_base()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    metadatas = [{"source": "knowledge_base.txt", "chunk": i} for i, _ in enumerate(chunks)]
    print(f"Loaded knowledge base and split it into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, make_embeddings(settings), metadatas=metadatas)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore: FAISS):
    """Build the LCEL RAG chain and return both chain and retriever."""

    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    llm = make_llm(settings)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    ).with_config({"run_name": "rag-chain-step1", "tags": ["rag", "step1"]})
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """Run one traced RAG query."""

    return chain.invoke(question)


def main() -> None:
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)
    print(f"Project: {settings.langsmith_project}")

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)
    questions = limit_items(SAMPLE_QUESTIONS, settings.question_limit)

    for i, question in enumerate(questions, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(questions):02d}] Q: {question}")
        print(f"       A: {answer[:240]}\n")

    print(f"Sent {len(questions)} traced RAG calls to LangSmith project '{settings.langsmith_project}'.")
    print("Open https://smith.langchain.com to verify input/output/latency traces.")


if __name__ == "__main__":
    main()
