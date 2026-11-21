"""The three RAG engines as LangGraph StateGraphs sharing one node library and one state schema.

  standard : retrieve -> generate
  hyde     : hypothesize(N) -> retrieve_multi -> generate          (Multi-Hypothesis HyDE)
  graph    : expand_query? -> link_entities -> hybrid_seed -> expand_subgraph <-> sufficiency_check (cyclic, k hops)
             -> fuse_rrf -> rerank? -> assemble_context -> generate

Standard and HyDE use the same chunks, embeddings, HNSW index and generation prompt as Graph RAG, so retrieval
strategy is the only experimental variable. Workflow toggles (UI tab 3) bypass optional nodes.
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
from src import prompts
from src.graph_store import normalize
from src.llm import UsageTracker, get_llm
from src.retrieval import get_index, rerank, rrf_fuse


# --------------------------------------------------------------------------- state
def _merge_dict(a: dict, b: dict) -> dict:
    """Reducer for per-stage timings: sum, so a node that runs several times (graph hop loop) accumulates."""
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0.0) + v
    return out


class RAGState(TypedDict, total=False):
    # inputs
    question: str
    practice_area: str
    hp: dict
    toggles: dict                         # {"query_expansion": bool, "rerank": bool, "graph": bool, ...}
    excluded_nodes: list[str]             # from the knowledge-graph canvas (tab 2)
    excluded_edges: list[str]
    # working
    hypotheses: list[str]
    query_entities: list[str]
    seed_nodes: list[str]
    frontier: list[str]
    visited_edges: list[str]
    hop: int
    triples: list[dict]
    rank_lists: dict[str, list[str]]
    contexts: list[dict]
    sufficient: bool
    # outputs
    answer: str
    trace: Annotated[list[dict], operator.add]
    timings: Annotated[dict, _merge_dict]   # ms per stage: retrieval / graph / llm
    usage: dict


def _areas(state: RAGState) -> set[str] | None:
    pa = state.get("practice_area", config.ALL_AREAS)
    return None if pa in (config.ALL_AREAS, None, "") else {pa}


def _hp(state: RAGState) -> config.Hyperparameters:
    base = config.DEFAULTS.as_dict()
    base.update(state.get("hp") or {})
    return config.Hyperparameters(**{k: base[k] for k in config.DEFAULTS.as_dict()})


def _on(state: RAGState, key: str, default: bool = True) -> bool:
    return bool((state.get("toggles") or {}).get(key, default))


def _timed(stage: str, state_update: dict, t0: float, bucket: str) -> dict:
    ms = (time.perf_counter() - t0) * 1000
    state_update["timings"] = {bucket: ms}
    state_update.setdefault("trace", []).insert(0, {"node": stage, "ms": round(ms, 1)})
    return state_update


class _Hyp(BaseModel):
    passages: list[str] = Field(description="One hypothetical passage per requested angle, in order")


class _Entities(BaseModel):
    entities: list[str] = Field(description="Named regulatory/contract entities in the question, full canonical names")


class _Sufficiency(BaseModel):
    sufficient: bool
    missing: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- node factories
def make_nodes(tracker: UsageTracker):
    cfg = {"callbacks": [tracker]}
    llm = get_llm(max_tokens=6000)   # structured calls: reasoning tokens count against the cap (HyDE failed twice at 1500)

    def _llm_time(t0: float) -> float:
        return (time.perf_counter() - t0) * 1000

    # ---- standard ----
    def retrieve(state: RAGState) -> dict:
        t0 = time.perf_counter()
        idx, hp = get_index(), _hp(state)
        hits = idx.vector_search(idx.embed_query(state["question"]), hp.top_k, _areas(state))
        ctx = [idx.context(i, s, "vector") for i, s in hits]
        return _timed("retrieve", {"contexts": ctx, "rank_lists": {"vector": [i for i, _ in hits]}}, t0, "retrieval")

    # ---- hyde ----
    def hypothesize(state: RAGState) -> dict:
        t0 = time.perf_counter()
        n = _hp(state).hyde_n
        angles = [prompts.HYDE_ANGLES[i % len(prompts.HYDE_ANGLES)] for i in range(n)]
        listing = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(angles))
        out = llm.with_structured_output(_Hyp).invoke(
            [("system", prompts.HYDE_SYSTEM),
             ("human", f"Question: {state['question']}\n\nWrite {n} short hypothetical passages (3-5 sentences each), "
                       f"one for each angle:\n{listing}")], config=cfg)
        passages = [p for p in (out.passages if out else []) if p.strip()][:n] or [state["question"]]
        return _timed("hypothesize", {"hypotheses": passages}, t0, "llm")

    def retrieve_multi(state: RAGState) -> dict:
        t0 = time.perf_counter()
        idx, hp = get_index(), _hp(state)
        lists, ranked = {}, {}
        for i, p in enumerate(state["hypotheses"], 1):
            # hypothetical passage is embedded as a document-like text (no query instruction prefix)
            hits = idx.vector_search(idx.embed_query(p, instruct=False), hp.top_k, _areas(state))
            lists[f"hyde_{i}"] = [c for c, _ in hits]
        fused = rrf_fuse(lists, hp.rrf_constant)[: hp.top_k]
        ctx = [idx.context(c, s, "+".join(src)) for c, s, src in fused]
        return _timed("retrieve_multi", {"contexts": ctx, "rank_lists": lists}, t0, "retrieval")

    # ---- graph ----
    def expand_query(state: RAGState) -> dict:
        """Optional Query Expansion node (workflow canvas): paraphrase the question in regulatory vocabulary."""
        t0 = time.perf_counter()
        out = llm.invoke([("system", prompts.HYDE_SYSTEM),
                          ("human", f"Rewrite this question as one precise regulatory-style search query (max 40 words), "
                                    f"keeping every named entity:\n{state['question']}")], config=cfg)
        return _timed("expand_query", {"hypotheses": [str(out.content).strip()]}, t0, "llm")

    def link_entities(state: RAGState) -> dict:
        t0 = time.perf_counter()
        idx = get_index()
        G = idx.graph.G
        out = llm.with_structured_output(_Entities).invoke(
            [("system", "Extract the named entities (regulators, provisions, instruments, organizations, controls, concepts) "
                        "mentioned or clearly implied by the question. Use full canonical names; expand defined terms."),
             ("human", state["question"])], config=cfg)
        names = [e for e in (out.entities if out else []) if e.strip()]
        t_llm = _llm_time(t0)
        from src.graph_store import EntityResolver

        resolver = EntityResolver(G, embed_fn=None)
        excl = set(state.get("excluded_nodes") or [])
        seeds, seen = [], set()
        t1 = time.perf_counter()
        for nm in names:
            nid, how = resolver.resolve(nm, "", "")
            if nid not in G:   # fall back to fuzzy over node names
                from rapidfuzz import process, fuzz

                best = process.extractOne(normalize(nm), list(G.nodes), scorer=fuzz.token_set_ratio, score_cutoff=85)
                nid = best[0] if best else None
            if nid and nid in G and nid not in excl and nid not in seen:
                seeds.append(nid)
                seen.add(nid)
        upd = {"query_entities": names, "seed_nodes": seeds, "frontier": seeds, "visited_edges": [], "hop": 0, "triples": []}
        upd["timings"] = {"llm": t_llm, "graph": (time.perf_counter() - t1) * 1000}
        upd["trace"] = [{"node": "link_entities", "ms": round(t_llm + (time.perf_counter() - t1) * 1000, 1), "entities": names, "linked": seeds}]
        return upd

    def hybrid_seed(state: RAGState) -> dict:
        t0 = time.perf_counter()
        idx, hp = get_index(), _hp(state)
        q = state["question"]
        qtext = " ".join([q] + (state.get("hypotheses") or [])[:1]) if _on(state, "query_expansion", False) else q
        vec = idx.vector_search(idx.embed_query(qtext), hp.top_k * 2, _areas(state))
        lists = {"vector": [c for c, _ in vec]}
        if _on(state, "bm25", True):   # sparse list; toggle off in the workflow canvas for vector-only
            lists["bm25"] = [c for c, _ in idx.bm25_search(qtext, hp.top_k * 2, _areas(state))]
        return _timed("hybrid_seed", {"rank_lists": lists}, t0, "retrieval")

    def expand_subgraph(state: RAGState) -> dict:
        t0 = time.perf_counter()
        idx = get_index()
        gs = idx.graph
        visited = set(state.get("visited_edges") or [])
        frontier = set(state.get("frontier") or [])
        new_edges, nxt = gs.expand(
            frontier, visited, _areas(state), set(state.get("excluded_nodes") or []), set(state.get("excluded_edges") or []),
            seed_vec=gs.centroid(set(state.get("seed_nodes") or [])))
        hop = state.get("hop", 0) + 1
        upd = {"frontier": sorted(nxt), "visited_edges": sorted(visited), "hop": hop, "triples": (state.get("triples") or []) + new_edges}
        out = _timed("expand_subgraph", upd, t0, "graph")
        out["trace"][0].update({"hop": hop, "new_edges": len(new_edges), "frontier": len(nxt)})
        return out

    def sufficiency_check(state: RAGState) -> dict:
        """Cyclic control: stop when the hop radius k is reached or there is nothing new to expand.
        Otherwise decide cheaply (no LLM) whether every question entity already has an edge to a retrieved chunk."""
        hp = _hp(state)
        hop, frontier = state.get("hop", 0), state.get("frontier") or []
        triples = state.get("triples") or []
        loop_off = not _on(state, "loop", True)   # workflow canvas: cyclic edge disabled -> single hop only
        done = hop >= hp.hop_radius or not frontier or (not triples and hop >= 1) or loop_off
        why = ("hop radius reached" if hop >= hp.hop_radius else "loop edge disabled" if loop_off
               else "frontier exhausted" if not frontier else "expand further")
        return {"sufficient": done, "trace": [{"node": "sufficiency_check", "hop": hop, "stop": done, "reason": why}]}

    def route_after_check(state: RAGState) -> Literal["expand_subgraph", "fuse_rrf"]:
        return "fuse_rrf" if state.get("sufficient") else "expand_subgraph"

    def fuse_rrf(state: RAGState) -> dict:
        t0 = time.perf_counter()
        idx, hp = get_index(), _hp(state)
        lists = dict(state.get("rank_lists") or {})
        # graph rank list: chunks backing the retrieved triples, ordered by hop discovery then multiplicity
        counts: dict[str, int] = {}
        order: list[str] = []
        for t in state.get("triples") or []:
            cid = t["chunk_id"]
            if cid not in counts:
                order.append(cid)
            counts[cid] = counts.get(cid, 0) + 1
        seed_chunks = {c for n in state.get("seed_nodes") or [] for c in idx.graph.G.nodes[n].get("chunk_ids", [])} if state.get("seed_nodes") else set()
        graph_ids = sorted(order, key=lambda c: (-(c in seed_chunks), -counts[c], order.index(c)))
        if graph_ids and _on(state, "graph", True):
            lists["graph"] = [c for c in graph_ids if c in idx.chunks]
        pool = max(hp.top_k * 2, 8) if _on(state, "rerank", True) else hp.top_k
        if not _on(state, "rrf", True):   # RRF bypassed: keep only the dense list in its own order
            lists = {"vector": lists.get("vector", [])}
        fused = rrf_fuse(lists, hp.rrf_constant)[:pool]
        ctx = [idx.context(c, s, "+".join(src)) for c, s, src in fused]
        return _timed("fuse_rrf", {"contexts": ctx}, t0, "retrieval")

    def rerank_node(state: RAGState) -> dict:
        t0 = time.perf_counter()
        hp = _hp(state)
        if _on(state, "rerank", True):
            ctx = rerank(state["question"], list(state["contexts"]), hp.top_k)
        else:
            ctx = list(state["contexts"])[: hp.top_k]
        return _timed("rerank" if _on(state, "rerank", True) else "rerank(bypassed)", {"contexts": ctx}, t0, "retrieval")

    def assemble_context(state: RAGState) -> dict:
        """Dedupe, attach the supporting triples (entity-relation grounding) to the contexts."""
        t0 = time.perf_counter()
        G = get_index().graph.G
        by_chunk: dict[str, list[str]] = {}
        for t in state.get("triples") or []:
            by_chunk.setdefault(t["chunk_id"], []).append(
                f"({G.nodes[t['u']].get('name', t['u'])}) -[{t['relation']}]-> ({G.nodes[t['v']].get('name', t['v'])})")
        ctx = []
        for c in state["contexts"]:
            c = dict(c)
            c["triples"] = by_chunk.get(c["chunk_id"], [])[:8]
            ctx.append(c)
        return _timed("assemble_context", {"contexts": ctx}, t0, "graph")

    # ---- shared generation ----
    def generate(state: RAGState) -> dict:
        t0 = time.perf_counter()
        ctx = state.get("contexts") or []
        extra = ""
        facts = [f"- {tr}" for c in ctx for tr in c.get("triples", [])]
        if facts:
            extra = "ENTITY-RELATION FACTS (from the knowledge graph; cite the source passage, not this list):\n" + "\n".join(dict.fromkeys(facts))
        msgs = [("system", prompts.build_system_prompt(state.get("practice_area", config.ALL_AREAS))),
                ("human", prompts.build_user_prompt(state["question"], ctx, extra))]
        res = get_llm(temperature=_hp(state).temperature, max_tokens=2000).invoke(msgs, config=cfg)
        return _timed("generate", {"answer": str(res.content), "usage": tracker.as_dict()}, t0, "llm")

    return dict(retrieve=retrieve, hypothesize=hypothesize, retrieve_multi=retrieve_multi, expand_query=expand_query,
                link_entities=link_entities, hybrid_seed=hybrid_seed, expand_subgraph=expand_subgraph,
                sufficiency_check=sufficiency_check, route_after_check=route_after_check, fuse_rrf=fuse_rrf,
                rerank_node=rerank_node, assemble_context=assemble_context, generate=generate)


# --------------------------------------------------------------------------- graph builders
def build_graph(engine: str, tracker: UsageTracker, toggles: dict | None = None):
    n = make_nodes(tracker)
    toggles = toggles or {}
    g = StateGraph(RAGState)
    if engine == "standard":
        g.add_node("retrieve", n["retrieve"])
        g.add_node("generate", n["generate"])
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
    elif engine == "hyde":
        g.add_node("hypothesize", n["hypothesize"])
        g.add_node("retrieve_multi", n["retrieve_multi"])
        g.add_node("generate", n["generate"])
        g.add_edge(START, "hypothesize")
        g.add_edge("hypothesize", "retrieve_multi")
        g.add_edge("retrieve_multi", "generate")
        g.add_edge("generate", END)
    elif engine == "graph":
        use_expand = toggles.get("query_expansion", False)
        use_graph = toggles.get("graph", True)
        g.add_node("link_entities", n["link_entities"])
        g.add_node("hybrid_seed", n["hybrid_seed"])
        g.add_node("expand_subgraph", n["expand_subgraph"])
        g.add_node("sufficiency_check", n["sufficiency_check"])
        g.add_node("fuse_rrf", n["fuse_rrf"])
        g.add_node("rerank", n["rerank_node"])
        g.add_node("assemble_context", n["assemble_context"])
        g.add_node("generate", n["generate"])
        if use_expand:
            g.add_node("expand_query", n["expand_query"])
            g.add_edge(START, "expand_query")
            g.add_edge("expand_query", "link_entities")
        else:
            g.add_edge(START, "link_entities")
        g.add_edge("link_entities", "hybrid_seed")
        if use_graph:
            g.add_edge("hybrid_seed", "expand_subgraph")
            g.add_edge("expand_subgraph", "sufficiency_check")
            g.add_conditional_edges("sufficiency_check", n["route_after_check"], {"expand_subgraph": "expand_subgraph", "fuse_rrf": "fuse_rrf"})
        else:
            g.add_edge("hybrid_seed", "fuse_rrf")   # vector + BM25 only
        g.add_edge("fuse_rrf", "rerank")
        g.add_edge("rerank", "assemble_context")
        g.add_edge("assemble_context", "generate")
        g.add_edge("generate", END)
    else:
        raise ValueError(engine)
    return g.compile()


@dataclass
class RunResult:
    engine: str
    question: str
    answer: str
    contexts: list[dict]
    trace: list[dict]
    timings: dict
    latency_ms: float
    usage: dict
    state: dict = field(default_factory=dict)


def run_engine(engine: str, question: str, practice_area: str = config.ALL_AREAS, hp: dict | None = None,
               toggles: dict | None = None, excluded_nodes: list[str] | None = None,
               excluded_edges: list[str] | None = None) -> RunResult:
    tracker = UsageTracker()
    app = build_graph(engine, tracker, toggles)
    t0 = time.perf_counter()
    out = app.invoke({"question": question, "practice_area": practice_area, "hp": hp or {}, "toggles": toggles or {},
                      "excluded_nodes": excluded_nodes or [], "excluded_edges": excluded_edges or [],
                      "timings": {}, "trace": []},
                     config={"recursion_limit": 60})
    latency = (time.perf_counter() - t0) * 1000
    return RunResult(engine, question, out.get("answer", ""), out.get("contexts", []), out.get("trace", []),
                     {k: round(v, 1) for k, v in (out.get("timings") or {}).items()}, round(latency, 1),
                     tracker.as_dict(), state=out)
