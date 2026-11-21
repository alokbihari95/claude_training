"""Fast unit tests: no LLM calls, no index required (the bge tokenizer is read from the local HF cache)."""
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import prompts
from src.evaluate import _ordered, aggregate, compare_to_spec
from src.graph_store import EntityResolver, ExtractedTriple, ChunkRef, GraphStore, normalize
from src.retrieval import rrf_fuse, tokenize


# ---- CO-STAR prompts -------------------------------------------------------------------------
@pytest.mark.parametrize("area", list(config.PRACTICE_AREAS))
def test_prompt_has_costar_and_only_own_guardrail(area):
    p = prompts.build_system_prompt(area)
    for section in ("# CONTEXT", "# OBJECTIVE", "# STYLE", "# TONE", "# AUDIENCE", "# RESPONSE FORMAT"):
        assert section in p
    assert "Senior Principal Advisory AI Assistant" in p
    assert "Gaps / Not found in corpus" in p
    assert config.PRACTICE_AREAS[area]["guardrails"] in p
    for other, v in config.PRACTICE_AREAS.items():
        if other != area:
            assert v["guardrails"] not in p


def test_all_areas_prompt_carries_every_guardrail():
    p = prompts.build_system_prompt(config.ALL_AREAS)
    for v in config.PRACTICE_AREAS.values():
        assert v["guardrails"] in p


# ---- RRF -------------------------------------------------------------------------------------
def test_rrf_formula_and_agreement_boost():
    fused = rrf_fuse({"a": ["x", "y"], "b": ["y", "z"]}, c=60)
    scores = {i: s for i, s, _ in fused}
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0][0] == "y"                      # appears in both lists
    assert dict((i, src) for i, _, src in fused)["y"] == ["a", "b"]


def test_rrf_constant_changes_weighting():
    lo = {i: s for i, s, _ in rrf_fuse({"a": ["x"]}, c=1)}
    hi = {i: s for i, s, _ in rrf_fuse({"a": ["x"]}, c=60)}
    assert lo["x"] > hi["x"]


def test_bm25_tokenizer_keeps_section_ids():
    assert "94z" in tokenize("Section 94Z applies") and "syn-cbdt/14/2025" in tokenize("Notification SYN-CBDT/14/2025")


# ---- chunking (600/60 tokens, bge tokenizer) ---------------------------------------------------
def test_chunker_respects_token_budget_and_overlap():
    from langchain_core.documents import Document
    from src.ingest import chunk_documents

    para = " ".join(f"Clause {i}: the deductor shall withhold tax at the prescribed rate." for i in range(120))
    doc = Document(page_content=para, metadata={"doc_id": "T-1", "title": "T", "practice_area": "tax"})
    chunks = chunk_documents([doc], 600, 60)
    assert len(chunks) > 1
    assert all(c["metadata"]["n_tokens"] <= 600 + 20 for c in chunks)       # + title prefix
    assert [c["chunk_id"] for c in chunks] == [f"T-1::c{i}" for i in range(len(chunks))]
    a, b = chunks[0]["text"], chunks[1]["text"]
    assert set(a.split()[-5:]) & set(b.split()[:60])                          # overlap present


# ---- entity resolution -----------------------------------------------------------------------
def _store_with(names):
    gs = GraphStore()
    res = EntityResolver(gs.G, embed_fn=None)
    gs.add_extraction(ChunkRef("D::c0", "D", "tax"), [(n, t) for n, t in names], [], res)
    return gs


def test_cbdt_aliases_merge_to_one_node():
    gs = _store_with([("CBDT", "Regulator"), ("Income Tax Department", "Regulator"), ("Central Board of Direct Taxes", "Regulator"),
                      ("Income Tax Dept", "Regulator"), ("Tax Authority", "Regulator")])
    assert list(gs.G.nodes) == ["cbdt"]
    assert {"Income Tax Department", "Tax Authority"} <= set(gs.G.nodes["cbdt"]["aliases"])


def test_digit_guard_prevents_false_merges():
    gs = _store_with([("Section 94Z", "Provision"), ("Section 94Z(5)", "Provision"), ("Section 271Z", "Provision")])
    assert len(gs.G.nodes) == 3


def test_literals_are_document_scoped_not_hubs():
    gs = GraphStore()
    res = EntityResolver(gs.G)
    gs.add_extraction(ChunkRef("A::c0", "A", "tax"), [("10%", "Value")], [], res)
    gs.add_extraction(ChunkRef("B::c0", "B", "grc"), [("10%", "Value")], [], res)
    assert len(gs.G.nodes) == 2                                                # '10%' in two docs must NOT become one bridge


def test_cross_document_entity_bridges_docs():
    gs = GraphStore()
    res = EntityResolver(gs.G)
    gs.add_extraction(ChunkRef("A::c0", "A", "tax"), [], [ExtractedTriple("Circular 7", "clarifies", "Section 94Z", "Instrument", "Provision")], res)
    gs.add_extraction(ChunkRef("B::c0", "B", "deals"), [], [ExtractedTriple("SPA", "covers", "Section 94Z", "Instrument", "Provision")], res)
    assert gs.G.nodes["section 94z"]["docs"] == ["A", "B"]


def test_expand_respects_exclusions_and_visited():
    gs = GraphStore()
    res = EntityResolver(gs.G)
    gs.add_extraction(ChunkRef("A::c0", "A", "tax"), [], [ExtractedTriple("X", "r", "Y", "Concept", "Concept"),
                                                          ExtractedTriple("Y", "r", "Z", "Concept", "Concept")], res)
    visited: set[str] = set()
    e1, f1 = gs.expand({"x"}, visited, excluded_nodes=set())
    assert {e["v"] for e in e1} == {"y"} and f1 == {"y"}
    e2, f2 = gs.expand(f1, visited)
    assert {e["v"] for e in e2} == {"z"}                                         # edge x-y already visited, not repeated
    e3, _ = gs.expand({"x"}, set(), excluded_nodes={"y"})
    assert e3 == []


# ---- node2vec ---------------------------------------------------------------------------------
def test_node2vec_shape_and_neighbours_closer():
    from src.node2vec import node2vec_embeddings

    G = nx.MultiDiGraph()
    for a, b in [("a", "b"), ("b", "c"), ("a", "c"), ("x", "y"), ("y", "z"), ("x", "z"), ("c", "x")]:
        G.add_edge(a, b)
    order, M = node2vec_embeddings(G, dim=16, epochs=15)
    assert M.shape == (6, 16)
    pos = {n: i for i, n in enumerate(order)}
    assert M[pos["a"]] @ M[pos["b"]] > M[pos["a"]] @ M[pos["z"]]


# ---- spec comparison logic ---------------------------------------------------------------------
def test_ordered_allows_ties_within_tolerance():
    assert _ordered({"a": 0.5, "b": 0.49, "c": 0.1}, desc=["a", "b", "c"])
    assert not _ordered({"a": 0.2, "b": 0.9, "c": 0.1}, desc=["a", "b", "c"])
    assert not _ordered({"a": 0.5, "b": None, "c": 0.1}, desc=["a", "b", "c"])


def _rec(engine, typ, rcr, faith, lat):
    return {"engine": engine, "type": typ, "rcr": rcr, "faithfulness": faith, "hallucination": 1 - faith, "correctness": rcr,
            "doc_recall": rcr, "has_citation": True, "latency_ms": lat, "timings": {"retrieval": 1, "graph": 1, "llm": 1},
            "usage": {"input_tokens": 1, "output_tokens": 1, "llm_calls": 1}, "error": None}


def test_compare_to_spec_flags_mismatch_not_hides_it():
    recs = []
    for t in config.QUESTION_TYPES:   # make Standard best everywhere: contradicts spec on multihop & vocab
        recs += [_rec("standard", t, 1.0, 0.9, 400), _rec("hyde", t, 0.5, 0.8, 1200), _rec("graph", t, 0.4, 0.95, 800)]
    cmp = compare_to_spec(aggregate(recs, config.ENGINES))
    assert cmp["simple"]["match"] is True
    assert cmp["multihop"]["match"] is False and cmp["vocab_mismatch"]["match"] is False
    assert cmp["latency"]["match"] is True
    assert cmp["hallucination"]["match"] is True


def test_normalize():
    assert normalize("The Company.") == "company" and normalize("  Section  94Z ") == "section 94z"
