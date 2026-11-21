"""LLM (OpenRouter) and embedding/reranker factories, plus token/latency accounting."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

import config


class UsageTracker(BaseCallbackHandler):
    """Accumulates token usage and LLM wall time for one pipeline run."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self._lock = threading.Lock()

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001
        with self._lock:
            self.llm_calls += 1
            for gens in response.generations:
                for g in gens:
                    um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                    self.input_tokens += um.get("input_tokens", 0)
                    self.output_tokens += um.get("output_tokens", 0)

    def as_dict(self) -> dict:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "llm_calls": self.llm_calls}


def get_llm(model: str | None = None, temperature: float = 0.0, max_tokens: int = 1500) -> ChatOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return ChatOpenAI(
        model=model or config.LLM_MODEL,
        api_key=key,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=3,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """bge-large-en-v1.5 (1024-dim), normalised so cosine == dot product."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        query_encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def warmup() -> None:
    """Load local models and open the LLM connection so the first timed query is not a cold start."""
    get_embeddings().embed_query("warm up")
    get_reranker().predict([("warm up", "warm up")], show_progress_bar=False)
    get_llm(max_tokens=5).invoke("ok")


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL, device="cpu", max_length=384)
