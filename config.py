"""Central configuration for the Standard vs HyDE vs Graph RAG benchmark."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
GRAPH_PATH = INDEX_DIR / "graph.json"
NODE2VEC_PATH = INDEX_DIR / "node2vec.npy"
EXTRACTION_CACHE = INDEX_DIR / "extraction_cache.json"
RESULTS_DIR = ROOT / "results"
GOLD_QA_PATH = DATA_DIR / "gold_qa.json"
GOLD_ENTITIES_PATH = DATA_DIR / "gold_entities.json"
SPEC_EXPECTATIONS_PATH = DATA_DIR / "spec_expectations.json"

# --- Models -----------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "anthropic/claude-haiku-5.5")      # generation, HyDE, extraction
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "anthropic/claude-sonnet-5.5")  # LLM-as-judge for KPIs
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"   # 1024-dim, 512-token limit
EMBEDDING_DIM = 1024
EMBEDDING_MAX_TOKENS = 512
RERANKER_MODEL = "BAAI/bge-reranker-base"

# --- Spec hyperparameters (the four configurable ones) ----------------------
@dataclass
class Hyperparameters:
    chunk_size: int = 600        # tokens
    chunk_overlap: int = 60      # tokens
    hyde_n: int = 3              # Multi-HyDE synthetic passages
    hop_radius: int = 2          # graph traversal k
    rrf_constant: int = 60       # RRF: 1 / (rrf_constant + rank)
    # Non-spec retrieval knobs (kept fixed across engines for fairness)
    top_k: int = 5
    temperature: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULTS = Hyperparameters()

# --- Practice areas (CO-STAR guardrails are built from these) ---------------
ALL_AREAS = "all"
PRACTICE_AREAS: dict[str, dict] = {
    "tax": {
        "label": "Tax & Regulatory Services",
        "focus": (
            "Direct Tax (CBDT notifications, Income Tax Act); Indirect Tax (CBIC circulars, "
            "57th GST Council rulings, ISD rules, e-invoicing)."
        ),
        "guardrails": (
            "You MUST cite the exact Section/Rule numbers for every statutory statement. "
            "Explicitly distinguish prospective amendments from retrospective clarifications."
        ),
    },
    "grc": {
        "label": "Risk Advisory & Governance (GRC)",
        "focus": (
            "RBI Master Directions (KYC/PMLA), SEBI BRSR guidelines, ISO 31000 risk management "
            "frameworks, internal audit controls."
        ),
        "guardrails": (
            "Flag non-compliance penalties immediately, at the top of the answer. "
            "Strictly map every control to the specific regulatory obligation it satisfies."
        ),
    },
    "deals": {
        "label": "Deal Advisory & Corporate Finance",
        "focus": (
            "M&A Share Purchase Agreements (SPA), Companies Act 2013 valuation standards, "
            "FEMA/FCRA foreign investment rules."
        ),
        "guardrails": (
            "Highlight indemnities, material adverse change (MAC) clauses, and financial "
            "exposure thresholds."
        ),
    },
}
AREA_CHOICES = [(v["label"], k) for k, v in PRACTICE_AREAS.items()] + [("All practice areas", ALL_AREAS)]

ENGINES = ["standard", "hyde", "graph"]
ENGINE_LABELS = {
    "standard": "Standard Vector RAG",
    "hyde": "Multi-Hypothesis HyDE RAG",
    "graph": "Graph RAG (LangGraph)",
}
QUESTION_TYPES = ["simple", "multihop", "vocab_mismatch"]
