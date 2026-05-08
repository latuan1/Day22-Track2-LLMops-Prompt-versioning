"""Step 2: Prompt Hub upload/pull plus deterministic A/B routing."""

from __future__ import annotations

import hashlib
from collections import Counter

from config import (
    EVIDENCE_DIR,
    configure_langsmith_tracing,
    ensure_directories,
    get_settings,
    limit_items,
    load_knowledge_base,
    make_embeddings,
    make_langsmith_client,
    make_llm,
)

settings = get_settings(require_openai=True, require_langsmith=True)
configure_langsmith_tracing(settings)
ensure_directories()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS


PROMPT_V1_NAME = settings.prompt_v1_name
PROMPT_V2_NAME = settings.prompt_v2_name

SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer the user's question using ONLY the "
    "provided context. Keep the answer concise, usually 2-4 sentences. If the "
    "context does not contain the answer, say: I don't have enough information.\n\n"
    "Context:\n{context}"
)
SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the facts relevant to the question.\n"
    "3. Write a clear, well-organized answer in 3-5 sentences.\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])


def push_prompts_to_hub(client) -> None:
    """Upload both prompt versions to LangSmith Prompt Hub."""

    uploads = [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 concise RAG answer prompt"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 structured RAG tutor prompt"),
    ]
    for name, prompt, description in uploads:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            print(f"Pushed '{name}' to Prompt Hub: {url}")
        except Exception as exc:
            print(f"Prompt Hub push skipped for '{name}': {exc}")


def pull_prompts_from_hub(client) -> dict[str, ChatPromptTemplate]:
    """Pull prompt versions from the Hub, falling back to local templates."""

    prompts: dict[str, ChatPromptTemplate] = {}
    for name, fallback in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"Pulled '{name}' from Prompt Hub")
        except Exception as exc:
            prompts[name] = fallback
            print(f"Using local fallback for '{name}': {exc}")
    return prompts


def get_prompt_version(request_id: str) -> str:
    """Route the same request_id to the same prompt version every time."""

    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def build_vectorstore() -> FAISS:
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


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict[str, str]:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


def main() -> None:
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)
    print(f"Project: {settings.langsmith_project}")
    print(f"Prompt names: {PROMPT_V1_NAME}, {PROMPT_V2_NAME}")

    client = make_langsmith_client(settings)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    llm = make_llm(settings)
    questions = limit_items(SAMPLE_QUESTIONS, settings.question_limit)
    counts: Counter[str] = Counter()
    log_lines: list[str] = []

    for i, question in enumerate(questions, 1):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        counts[version_tag] += 1
        result = ask_ab(retriever, llm, prompts[version_key], question, version_tag)
        line = f"[{i:02d}] [prompt-{version_tag}] {request_id} | {question} | {result['answer'][:160]}"
        print(line)
        log_lines.append(line)

    summary = f"Routing summary: prompt-v1={counts['v1']}, prompt-v2={counts['v2']}"
    print(summary)
    log_lines.append(summary)
    log_path = EVIDENCE_DIR / "02_ab_routing_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Saved routing log to {log_path}")


if __name__ == "__main__":
    main()
