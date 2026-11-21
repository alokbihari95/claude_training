"""Knowledge-graph store (NetworkX MultiDiGraph) with entity normalisation and k-hop traversal.

Neo4j can replace this behind the same small interface (add_extraction / expand / neighbours).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
from rapidfuzz import fuzz

import config

LITERAL_TYPES = {"value", "date", "amount", "rate", "period"}   # scoped to one document, never merged
_DIGIT_TOK = re.compile(r"\d")


def normalize(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^\w\s/%\-\(\)\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s


def _digit_tokens(s: str) -> frozenset[str]:
    return frozenset(t for t in re.split(r"\s+", s) if _DIGIT_TOK.search(t))


def edge_id(u: str, rel: str, v: str, chunk_id: str) -> str:
    return f"{u}|{rel}|{v}|{chunk_id}"


@dataclass
class ExtractedTriple:
    subject: str
    relation: str
    object: str
    subject_type: str = ""
    object_type: str = ""


@dataclass
class ChunkRef:
    chunk_id: str
    doc_id: str
    practice_area: str


class EntityResolver:
    """Maps surface forms to canonical node ids: seed alias table, exact, fuzzy, then embedding similarity."""

    def __init__(self, G: nx.MultiDiGraph, embed_fn=None, fuzzy_threshold: int = 92, embed_threshold: float = 0.90):
        self.G = G
        self.embed_fn = embed_fn
        self.fuzzy_threshold = fuzzy_threshold
        self.embed_threshold = embed_threshold
        self.seed: dict[str, str] = {}     # normalised alias -> normalised canonical
        seed_path = config.DATA_DIR / "alias_seed.json"
        if seed_path.exists():
            for canon, aliases in json.loads(seed_path.read_text()).items():
                if canon.startswith("_"):
                    continue
                c = normalize(canon)
                self.seed[c] = c
                for a in aliases:
                    self.seed[normalize(a)] = c
        self.alias_index: dict[str, str] = {}   # normalised surface form -> node id
        self._vec: dict[str, np.ndarray] = {}
        for nid, d in G.nodes(data=True):
            for a in d.get("aliases", []) + [d.get("name", nid)]:
                self.alias_index[normalize(a)] = nid

    def _embed(self, text: str) -> np.ndarray | None:
        """`embed_fn` is a batch function (list[str] -> list[vector]); missing vectors are embedded together."""
        if self.embed_fn is None:
            return None
        if text not in self._vec:
            self._ensure([text])
        return self._vec[text]

    def _ensure(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self._vec]
        if missing:
            for t, v in zip(missing, self.embed_fn(missing)):
                self._vec[t] = np.asarray(v, dtype=np.float32)

    def resolve(self, surface: str, etype: str, doc_id: str) -> tuple[str, str]:
        """Return (node_id, method). Creates nothing; caller adds the node if it is new."""
        norm = normalize(surface)
        if etype.lower() in LITERAL_TYPES:
            return f"{norm}@{doc_id}", "literal"
        norm = self.seed.get(norm, norm)
        if norm in self.alias_index:
            return self.alias_index[norm], "exact"
        digits = _digit_tokens(norm)
        best, best_score = None, 0.0
        for nid, d in self.G.nodes(data=True):
            if d.get("type", "").lower() in LITERAL_TYPES:
                continue
            for cand in {nid, *[normalize(a) for a in d.get("aliases", [])]}:
                if _digit_tokens(cand) != digits:
                    continue
                sc = fuzz.token_set_ratio(norm, cand)
                # token_set_ratio is 100 for a pure subset ("meridian" vs "meridian finance ltd"); require length parity
                if sc >= self.fuzzy_threshold and abs(len(norm) - len(cand)) <= max(4, 0.35 * len(cand)) and sc > best_score:
                    best, best_score = nid, sc
        if best:
            return best, "fuzzy"
        if self.embed_fn is not None and len(norm) > 3:
            cands = [
                (nid, d) for nid, d in self.G.nodes(data=True)
                if d.get("type", "").lower() not in LITERAL_TYPES and _digit_tokens(nid) == digits
                and not (etype and d.get("type") and etype.lower() != d["type"].lower())
                and fuzz.token_set_ratio(norm, nid) >= 45   # lexical prefilter: only embed plausible candidates (bge on CPU is the slow part)
            ]
            self._ensure([norm] + [nid for nid, _ in cands])
            v = self._vec[norm]
            best_sim = 0.0
            for nid, d in cands:
                sim = float(np.dot(v, self._vec[nid]))
                if sim >= self.embed_threshold and sim > best_sim:
                    best, best_sim = nid, sim
            if best:
                return best, "embedding"
        return norm, "new"


class GraphStore:
    def __init__(self) -> None:
        self.G = nx.MultiDiGraph()
        self.node_order: list[str] = []
        self.node2vec: np.ndarray | None = None
        self.resolution_log: list[dict] = []   # surface -> node id, for resolution accuracy

    # -- construction -------------------------------------------------------
    def add_extraction(
        self, chunk: ChunkRef, entities: list[tuple[str, str]], triples: list[ExtractedTriple], resolver: EntityResolver
    ) -> None:
        def ensure(surface: str, etype: str) -> str:
            nid, method = resolver.resolve(surface, etype, chunk.doc_id)
            if nid not in self.G:
                self.G.add_node(nid, name=surface.strip(), type=etype or "Concept", aliases=[], docs=[], chunk_ids=[], areas=[])
            d = self.G.nodes[nid]
            if surface.strip() not in d["aliases"] and surface.strip() != d["name"]:
                d["aliases"].append(surface.strip())
            if chunk.doc_id not in d["docs"]:
                d["docs"].append(chunk.doc_id)
            if chunk.chunk_id not in d["chunk_ids"]:
                d["chunk_ids"].append(chunk.chunk_id)
            if chunk.practice_area not in d["areas"]:
                d["areas"].append(chunk.practice_area)
            resolver.alias_index[normalize(surface)] = nid
            resolver.alias_index[nid] = nid
            self.resolution_log.append({"surface": surface.strip(), "node": nid, "method": method, "doc": chunk.doc_id})
            return nid

        for name, etype in entities:
            if name.strip():
                ensure(name, etype)
        for t in triples:
            if not (t.subject.strip() and t.object.strip() and t.relation.strip()):
                continue
            u = ensure(t.subject, t.subject_type)
            v = ensure(t.object, t.object_type)
            if u == v:
                continue
            rel = t.relation.strip().lower()
            eid = edge_id(u, rel, v, chunk.chunk_id)
            if not self.G.has_edge(u, v, key=eid):
                self.G.add_edge(u, v, key=eid, relation=rel, chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, area=chunk.practice_area)

    # -- persistence --------------------------------------------------------
    def save(self, path: Path = config.GRAPH_PATH, vec_path: Path = config.NODE2VEC_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.G, edges="edges")
        path.write_text(json.dumps({"graph": data, "node_order": self.node_order, "resolution_log": self.resolution_log}))
        if self.node2vec is not None:
            np.save(vec_path, self.node2vec)

    @classmethod
    def load(cls, path: Path = config.GRAPH_PATH, vec_path: Path = config.NODE2VEC_PATH) -> "GraphStore":
        gs = cls()
        if path.exists():
            raw = json.loads(path.read_text())
            gs.G = nx.node_link_graph(raw["graph"], directed=True, multigraph=True, edges="edges")
            gs.node_order = raw.get("node_order", list(gs.G.nodes))
            gs.resolution_log = raw.get("resolution_log", [])
        if vec_path.exists():
            gs.node2vec = np.load(vec_path)
        return gs

    def compute_node2vec(self, **kw) -> None:
        from src.node2vec import node2vec_embeddings

        self.node_order, self.node2vec = node2vec_embeddings(self.G, **kw)

    # -- traversal ----------------------------------------------------------
    def incident_edges(self, node: str):
        """All edges touching `node`, direction-agnostic: yields (eid, u, v, data)."""
        for u, v, k, d in self.G.out_edges(node, keys=True, data=True):
            yield k, u, v, d
        for u, v, k, d in self.G.in_edges(node, keys=True, data=True):
            yield k, u, v, d

    def expand(
        self,
        frontier: set[str],
        visited_edges: set[str],
        areas: set[str] | None = None,
        excluded_nodes: set[str] | None = None,
        excluded_edges: set[str] | None = None,
        max_frontier: int = 40,
        seed_vec: np.ndarray | None = None,
    ) -> tuple[list[dict], set[str]]:
        """One hop of expansion. Returns (new edge records, next frontier)."""
        excluded_nodes = excluded_nodes or set()
        excluded_edges = excluded_edges or set()
        new_edges: list[dict] = []
        nxt: set[str] = set()
        for node in frontier:
            for eid, u, v, d in self.incident_edges(node):
                if eid in visited_edges or eid in excluded_edges:
                    continue
                if u in excluded_nodes or v in excluded_nodes:
                    continue
                if areas and d.get("area") not in areas:
                    continue
                visited_edges.add(eid)
                other = v if u == node else u
                new_edges.append({"eid": eid, "u": u, "v": v, "relation": d["relation"], "chunk_id": d["chunk_id"], "doc_id": d["doc_id"]})
                nxt.add(other)
        if seed_vec is not None and self.node2vec is not None and len(nxt) > max_frontier:
            pos = {n: i for i, n in enumerate(self.node_order)}
            nxt = set(sorted(nxt, key=lambda n: -float(self.node2vec[pos[n]] @ seed_vec) if n in pos else 0.0)[:max_frontier])
        return new_edges, nxt

    def centroid(self, nodes: set[str]) -> np.ndarray | None:
        if self.node2vec is None or not nodes:
            return None
        pos = {n: i for i, n in enumerate(self.node_order)}
        rows = [self.node2vec[pos[n]] for n in nodes if n in pos]
        if not rows:
            return None
        c = np.mean(rows, axis=0)
        return c / (np.linalg.norm(c) + 1e-9)

    def stats(self) -> dict:
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(),
                "docs": len({d for _, x in self.G.nodes(data=True) for d in x.get("docs", [])})}
