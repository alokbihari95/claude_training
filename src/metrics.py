"""The four spec KPIs (+ supporting metrics).

1. Retrieval Context Recall (RCR)      judge: which gold atomic facts are present in the retrieved chunks
2. Faithfulness / Hallucination score  judge: answer claims supported by the cited sources (hallucination = 1 - faithfulness)
3. Entity-Relation Resolution Accuracy graph index vs data/gold_entities.json (a property of the graph; n/a for Standard / HyDE)
4. End-to-end latency                  retrieval / graph traversal / LLM inference breakdown (recorded by the engines)
Supporting: answer correctness (judge), doc recall (gold docs among retrieved), citation presence.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import networkx as nx
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

import config
from src.graph_store import GraphStore, normalize
from src.llm import UsageTracker, get_llm


# --------------------------------------------------------------------------- judge schemas
class _FactCheck(BaseModel):
    present: list[bool] = Field(description="For each numbered fact, in order: is it stated in the passages?")


class _Claim(BaseModel):
    claim: str
    supported: bool


class _AnswerJudgement(BaseModel):
    claims: list[_Claim] = Field(description="Atomic factual claims in the answer (exclude headings, hedges and 'not found' statements)")
    correctness: float = Field(description="0.0-1.0 agreement with the reference answer on the key facts")


RCR_SYSTEM = ("You are a strict evaluator. Given retrieved passages and a numbered list of facts, decide for each fact whether the passages "
              "explicitly contain the information (same figures, names, periods). Paraphrase is fine; inference from unrelated text is not.")
ANSWER_SYSTEM = ("You are a strict evaluator of regulatory answers. (1) Split the ANSWER into atomic factual claims and mark each as supported "
                 "only if the SOURCES explicitly support it (numbers, dates, names must match). Ignore formatting, hedges, and statements that "
                 "something is not found. (2) Rate correctness: the share of the REFERENCE's key facts that the answer states correctly "
                 "(0.0-1.0; penalise contradictions).")


def _src_text(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{c['metadata'].get('doc_id')}] {c['text']}" for c in contexts)


def judge_rcr(judge, question: dict, contexts: list[dict], tracker: UsageTracker | None = None) -> float:
    facts = question["facts"]
    listing = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    out = judge.with_structured_output(_FactCheck).invoke(
        [("system", RCR_SYSTEM), ("human", f"PASSAGES\n{_src_text(contexts)}\n\nFACTS\n{listing}")],
        config={"callbacks": [tracker]} if tracker else None)
    flags = (out.present if out else [])[: len(facts)]
    flags += [False] * (len(facts) - len(flags))
    return sum(flags) / len(facts)


def judge_answer(judge, question: dict, answer: str, contexts: list[dict], tracker: UsageTracker | None = None) -> dict:
    out = judge.with_structured_output(_AnswerJudgement).invoke(
        [("system", ANSWER_SYSTEM),
         ("human", f"QUESTION\n{question['question']}\n\nREFERENCE\n{question['gold_answer']}\n\nSOURCES\n{_src_text(contexts)}\n\nANSWER\n{answer}")],
        config={"callbacks": [tracker]} if tracker else None)
    claims = out.claims if out else []
    faith = (sum(c.supported for c in claims) / len(claims)) if claims else 1.0   # no claims -> nothing unsupported
    return {"faithfulness": round(faith, 4), "hallucination": round(1 - faith, 4), "n_claims": len(claims),
            "unsupported": [c.claim for c in claims if not c.supported],
            "correctness": round(min(max(out.correctness if out else 0.0, 0.0), 1.0), 4)}


def doc_recall(question: dict, contexts: list[dict]) -> float:
    got = {c["metadata"].get("doc_id") for c in contexts}
    gold = set(question["gold_docs"])
    return len(gold & got) / len(gold)


_CITE = re.compile(r"\[[A-Z]+-\d+")


def has_citation(answer: str) -> bool:
    return bool(_CITE.search(answer))


# --------------------------------------------------------------------------- ER resolution accuracy
def _find_nodes(gs: GraphStore, surface_map: dict[str, set[str]], name: str) -> set[str]:
    n = normalize(name)
    if n in surface_map:
        return surface_map[n]
    return set()


def er_resolution_accuracy(gs: GraphStore | None = None) -> dict:
    """Does every surface form of a gold entity resolve to ONE node, without absorbing other gold entities?

    group_accuracy : gold groups (>=1 alias seen in the corpus) whose seen aliases all map to a single node that
                     contains no alias of any other gold group
    alias_coverage : gold aliases that the graph actually saw as a surface form (informational)
    link_recall    : gold cross-document links whose endpoints are connected within 2 undirected hops
    """
    gs = gs or GraphStore.load()
    gold = json.loads(config.GOLD_ENTITIES_PATH.read_text())
    surface_map: dict[str, set[str]] = {}
    for r in gs.resolution_log:
        surface_map.setdefault(normalize(r["surface"]), set()).add(r["node"])
    for nid, d in gs.G.nodes(data=True):
        surface_map.setdefault(normalize(d.get("name", nid)), set()).add(nid)
        surface_map.setdefault(nid, set()).add(nid)

    alias_to_group = {normalize(a): g["canonical"] for g in gold["entities"] for a in g["aliases"]}
    groups_seen, groups_ok, alias_total, alias_seen, detail = 0, 0, 0, 0, []
    for g in gold["entities"]:
        nodes_by_alias = {a: _find_nodes(gs, surface_map, a) for a in g["aliases"]}
        alias_total += len(g["aliases"])
        seen = {a: ns for a, ns in nodes_by_alias.items() if ns}
        alias_seen += len(seen)
        if not seen:
            continue
        groups_seen += 1
        all_nodes = set().union(*seen.values())
        single = len(all_nodes) == 1
        clean = True
        if single:
            node = next(iter(all_nodes))
            d = gs.G.nodes[node]
            forms = {normalize(x) for x in d.get("aliases", [])} | {normalize(d.get("name", node)), node}
            others = {alias_to_group[f] for f in forms if f in alias_to_group and alias_to_group[f] != g["canonical"]}
            clean = not others
        ok = single and clean
        groups_ok += ok
        detail.append({"entity": g["canonical"], "ok": ok, "nodes": sorted(all_nodes)})

    und = nx.Graph(gs.G.to_undirected(as_view=False))

    def best_node(name: str) -> set[str]:
        ns = _find_nodes(gs, surface_map, name)
        if ns:
            return ns
        n = normalize(name)
        scored = [(fuzz.token_set_ratio(n, nid), nid) for nid in gs.G.nodes]
        scored = [s for s in scored if s[0] >= 88]
        return {max(scored)[1]} if scored else set()

    links_ok = 0
    for L in gold["links"]:
        s, o = best_node(L["subject"]), best_node(L["object"])
        ok = False
        for a in s:
            for b in o:
                if a in und and b in und and nx.has_path(und, a, b) and nx.shortest_path_length(und, a, b) <= 2:
                    ok = True
        links_ok += ok
    return {"group_accuracy": round(groups_ok / groups_seen, 4) if groups_seen else 0.0, "groups_evaluated": groups_seen,
            "groups_correct": groups_ok, "alias_coverage": round(alias_seen / alias_total, 4),
            "link_recall": round(links_ok / len(gold["links"]), 4), "links": len(gold["links"]), "detail": detail}
