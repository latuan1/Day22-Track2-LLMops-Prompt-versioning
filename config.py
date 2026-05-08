"""Shared configuration helpers for the Day 22 LangSmith/RAG lab."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.txt"


load_dotenv(PROJECT_ROOT / ".env")


T = TypeVar("T")


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Expected an integer value, got {value!r}") from exc
    return parsed if parsed > 0 else None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-") or "day22"


@dataclass(frozen=True)
class Settings:
    langsmith_api_key: str | None
    langsmith_endpoint: str
    langsmith_project: str
    langsmith_tracing: bool
    openai_api_key: str | None
    openai_base_url: str | None
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    question_limit: int | None
    prompt_prefix: str
    prompt_v1_name: str
    prompt_v2_name: str


def get_settings(
    *,
    require_openai: bool = False,
    require_langsmith: bool = False,
) -> Settings:
    """Load settings from .env/environment and optionally validate secrets."""

    langsmith_project = (
        _first_env(
            "LANGSMITH_PROJECT",
            "LANGCHAIN_PROJECT",
            default="day22-langsmith-prompt-versioning",
        )
        or "day22-langsmith-prompt-versioning"
    )
    prompt_prefix = _first_env("PROMPT_PREFIX", default=_slug(langsmith_project)) or _slug(
        langsmith_project
    )

    settings = Settings(
        langsmith_api_key=_first_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        langsmith_endpoint=_first_env(
            "LANGSMITH_ENDPOINT",
            "LANGCHAIN_ENDPOINT",
            default="https://api.smith.langchain.com",
        )
        or "https://api.smith.langchain.com",
        langsmith_project=langsmith_project,
        langsmith_tracing=_truthy(
            _first_env("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", default="true")
        ),
        openai_api_key=_first_env("OPENAI_API_KEY", "OPENAI_KEY"),
        openai_base_url=_first_env(
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_ENDPOINT",
            "OPENAI_API_ENDPOINT",
        ),
        llm_model=_first_env(
            "OPENAI_MODEL",
            "DEFAULT_LLM_MODEL",
            "LLM_MODEL",
            default="gpt-5.4-mini",
        )
        or "gpt-5.4-mini",
        embedding_model=_first_env(
            "OPENAI_EMBEDDING_MODEL",
            "EMBEDDING_MODEL",
            default="text-embedding-3-small",
        )
        or "text-embedding-3-small",
        chunk_size=int(_first_env("RAG_CHUNK_SIZE", default="500") or "500"),
        chunk_overlap=int(_first_env("RAG_CHUNK_OVERLAP", default="50") or "50"),
        retrieval_k=int(_first_env("RAG_RETRIEVAL_K", default="3") or "3"),
        question_limit=_optional_int(_first_env("LAB_QUESTION_LIMIT")),
        prompt_prefix=prompt_prefix,
        prompt_v1_name=_first_env(
            "PROMPT_V1_NAME", default=f"{prompt_prefix}-rag-prompt-v1"
        )
        or f"{prompt_prefix}-rag-prompt-v1",
        prompt_v2_name=_first_env(
            "PROMPT_V2_NAME", default=f"{prompt_prefix}-rag-prompt-v2"
        )
        or f"{prompt_prefix}-rag-prompt-v2",
    )

    missing: list[str] = []
    if require_openai and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if require_langsmith and not settings.langsmith_api_key:
        missing.append("LANGSMITH_API_KEY")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variable(s): {joined}")

    return settings


def configure_langsmith_tracing(settings: Settings) -> None:
    """Set both current LangSmith and legacy LangChain tracing variables."""

    tracing_value = "true" if settings.langsmith_tracing else "false"
    os.environ["LANGSMITH_TRACING"] = tracing_value
    os.environ["LANGCHAIN_TRACING_V2"] = tracing_value
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key


def make_llm(settings: Settings, *, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": settings.llm_model,
        "api_key": settings.openai_api_key,
        "temperature": temperature,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def make_embeddings(settings: Settings):
    from langchain_openai import OpenAIEmbeddings

    kwargs = {
        "model": settings.embedding_model,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


def make_langsmith_client(settings: Settings):
    from langsmith import Client

    kwargs = {"api_key": settings.langsmith_api_key}
    if settings.langsmith_endpoint:
        kwargs["api_url"] = settings.langsmith_endpoint
    try:
        return Client(**kwargs)
    except TypeError:
        kwargs.pop("api_url", None)
        return Client(**kwargs)


def load_knowledge_base() -> str:
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}")
    return KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8")


def limit_items(items: Sequence[T], limit: int | None) -> list[T]:
    return list(items[:limit]) if limit else list(items)


def print_config_summary(settings: Settings) -> None:
    print("Config loaded successfully")
    print(f"   LangSmith project : {settings.langsmith_project}")
    print(f"   LangSmith endpoint: {settings.langsmith_endpoint}")
    print(f"   Tracing enabled   : {settings.langsmith_tracing}")
    print(f"   OpenAI endpoint   : {settings.openai_base_url or 'default OpenAI API'}")
    print(f"   Default LLM model : {settings.llm_model}")
    print(f"   Embedding model   : {settings.embedding_model}")
    print(f"   Retrieval k       : {settings.retrieval_k}")
    print(f"   Prompt V1 name    : {settings.prompt_v1_name}")
    print(f"   Prompt V2 name    : {settings.prompt_v2_name}")
    if settings.question_limit:
        print(f"   Question limit    : {settings.question_limit}")


def ensure_directories(paths: Iterable[Path] = (DATA_DIR, EVIDENCE_DIR)) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = get_settings(require_openai=True, require_langsmith=True)
    configure_langsmith_tracing(cfg)
    ensure_directories()
    print_config_summary(cfg)
