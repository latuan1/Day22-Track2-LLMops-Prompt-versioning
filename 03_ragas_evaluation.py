"""Step 3: Evaluate both RAG prompt versions with RAGAS."""

from __future__ import annotations

import json
import math
import shutil
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np

from config import (
    DATA_DIR,
    EVIDENCE_DIR,
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

try:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
except ImportError:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import QA_PAIRS


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

PROMPTS = {
    "v1": ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")]),
    "v2": ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")]),
}

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


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


@traceable(name="ragas-rag-query", tags=["ragas", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict[str, list[str] | str]:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context_text, "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore: FAISS, prompt_version: str) -> list[dict]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    llm = make_llm(settings)
    prompt = PROMPTS[prompt_version]
    qa_pairs = limit_items(QA_PAIRS, settings.question_limit)
    results: list[dict] = []

    print(f"\nRunning {len(qa_pairs)} questions with prompt {prompt_version} ...")
    for i, qa in enumerate(qa_pairs, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/{len(qa_pairs):02d}] {qa['question']}")
    return results


def build_ragas_dataset(rag_results: list[dict]):
    samples = [
        SingleTurnSample(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
            reference=item["reference"],
        )
        for item in rag_results
    ]
    return EvaluationDataset(samples=samples)


def _metric_values(result, metric_name: str) -> list[float]:
    try:
        raw = result[metric_name]
    except Exception:
        if hasattr(result, "to_pandas"):
            raw = result.to_pandas()[metric_name].tolist()
        else:
            raise

    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, (int, float)):
        raw = [raw]

    values: list[float] = []
    for value in raw:
        if value is None:
            continue
        score = float(value)
        if not math.isnan(score):
            values.append(score)
    return values


def run_ragas_eval(rag_results: list[dict], version: str) -> dict[str, float]:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=make_llm(settings),
        embeddings=make_embeddings(settings),
    )

    scores: dict[str, float] = {}
    for metric_name in METRIC_NAMES:
        values = _metric_values(result, metric_name)
        scores[metric_name] = float(np.mean(values)) if values else 0.0
        flag = " target" if metric_name == "faithfulness" and scores[metric_name] >= 0.8 else ""
        print(f"  {metric_name:20s}: {scores[metric_name]:.4f}{flag}")
    return scores


def _print_comparison(v1_scores: dict[str, float], v2_scores: dict[str, float]) -> None:
    print("\nRAGAS comparison")
    print("-" * 60)
    print(f"{'Metric':24s} {'V1':>10s} {'V2':>10s} {'Winner':>10s}")
    print("-" * 60)
    for metric_name in METRIC_NAMES:
        s1 = v1_scores[metric_name]
        s2 = v2_scores[metric_name]
        winner = "V1" if s1 > s2 else "V2" if s2 > s1 else "Tie"
        print(f"{metric_name:24s} {s1:10.4f} {s2:10.4f} {winner:>10s}")
    print("-" * 60)


def main() -> None:
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    _print_comparison(v1_scores, v2_scores)
    best_faithfulness = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    target_met = best_faithfulness >= 0.8
    if target_met:
        print(f"Target met: faithfulness = {best_faithfulness:.4f}")
    else:
        print(f"Below target: faithfulness = {best_faithfulness:.4f}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(v1_results),
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "best_faithfulness": best_faithfulness,
        "target_met": target_met,
        "settings": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "retrieval_k": settings.retrieval_k,
        },
    }
    report_path = DATA_DIR / "ragas_report.json"
    evidence_path = EVIDENCE_DIR / "03_ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copyfile(report_path, evidence_path)
    print(f"Saved {report_path}")
    print(f"Copied report to {evidence_path}")


if __name__ == "__main__":
    main()
