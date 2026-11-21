"""Tab 3: visual execution DAG drawn from the *compiled* LangGraph, with per-node toggles.

Layout comes from the real compiled graph (build_graph(...).get_graph()), so the picture cannot drift from what runs.
Nodes that a toggle bypasses are drawn dashed and grey.
"""
from __future__ import annotations

import html

from src.engines import build_graph
from src.llm import UsageTracker

# toggle key -> (label shown in UI, compiled node ids it controls)
TOGGLES = {
    "query_expansion": ("Query Expansion", ["expand_query"]),
    "bm25": ("BM25 sparse search", []),
    "graph": ("Graph traversal (k-hop)", ["expand_subgraph", "sufficiency_check"]),
    "loop": ("Cyclic loop edge (multi-hop)", []),
    "rrf": ("RRF Fusion", ["fuse_rrf"]),
    "rerank": ("Rerank", ["rerank"]),
}
DEFAULT_TOGGLES = {"query_expansion": False, "bm25": True, "graph": True, "loop": True, "rrf": True, "rerank": True}
SPEC_FLOW = [("Query Expansion", "query_expansion"), ("Hybrid Vector/Graph Search", "graph"), ("RRF Fusion", "rrf"), ("Rerank", "rerank"), ("Synthesis", None)]
LABELS = {"__start__": "START", "__end__": "END", "expand_query": "Query Expansion", "link_entities": "Entity Linking",
          "hybrid_seed": "Hybrid Seed\n(BM25 + HNSW)", "expand_subgraph": "Expand Subgraph\n(k-hop)", "sufficiency_check": "Sufficiency\nCheck",
          "fuse_rrf": "RRF Fusion", "rerank": "Rerank", "assemble_context": "Assemble Context", "generate": "Synthesis\n(CO-STAR)"}
W, H, GX, GY = 150, 44, 40, 62


def _layers(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, int]:
    """Longest-path depth ignoring back edges (the sufficiency -> expand loop)."""
    depth = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        changed = False
        for s, t in edges:
            if (s, t) == ("sufficiency_check", "expand_subgraph"):
                continue
            if depth[t] < depth[s] + 1:
                depth[t] = depth[s] + 1
                changed = True
        if not changed:
            break
    return depth


def workflow_svg(toggles: dict) -> str:
    full = build_graph("graph", UsageTracker(), {**DEFAULT_TOGGLES, "query_expansion": True}).get_graph()
    live_nodes = set(build_graph("graph", UsageTracker(), toggles).get_graph().nodes)
    nodes = list(full.nodes)
    edges = [(e.source, e.target) for e in full.edges]
    depth = _layers(nodes, edges)
    # the loop edge is only a loop if both nodes are live
    rows: dict[int, list[str]] = {}
    for n in nodes:
        rows.setdefault(depth[n], []).append(n)
    pos = {}
    maxw = max(len(v) for v in rows.values())
    for d, ns in rows.items():
        for i, n in enumerate(ns):
            pos[n] = ((i + (maxw - len(ns)) / 2) * (W + GX) + 110, d * (H + GY) + 20)   # 110px left margin for the loop edge
    width = maxw * (W + GX) + 130
    height = (max(rows) + 1) * (H + GY) + 30
    passthrough = {n for k, (_, ns) in TOGGLES.items() if not toggles.get(k, DEFAULT_TOGGLES[k]) for n in ns}   # compiled but bypassed at runtime
    off = lambda n: n not in live_nodes or n in passthrough  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;background:#fcfcfb;border:1px solid #e1e0da;border-radius:8px">',
             '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>']
    for s, t in edges:
        (x0, y0), (x1, y1) = pos[s], pos[t]
        dead = off(s) or off(t)
        back = (s, t) == ("sufficiency_check", "expand_subgraph")
        sx, sy, tx, ty = x0 + W / 2, y0 + H, x1 + W / 2, y1
        col, dash = ("#b9b8b2", "5,4") if dead else ("#52514e", "")
        if back:
            sx, sy, tx, ty = x0, y0 + H / 2, x1, y1 + H / 2
            d = f"M{sx},{sy} C{sx - 70},{sy} {tx - 70},{ty} {tx},{ty}"
            if toggles.get("loop", True) is False:
                col, dash = "#b9b8b2", "5,4"
        elif abs(sx - tx) < 1:
            d = f"M{sx},{sy} L{tx},{ty}"
        else:
            my = (sy + ty) / 2
            d = f"M{sx},{sy} C{sx},{my} {tx},{my} {tx},{ty}"
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6" stroke-dasharray="{dash}" marker-end="url(#ar)"/>')
    for n in nodes:
        x, y = pos[n]
        lab = LABELS.get(n, n).split("\n")
        special = n in ("__start__", "__end__")
        fill = "#f1f0ec" if special else ("#f6f6f4" if off(n) else "#e8f0fb")
        stroke, dash, txt = ("#b9b8b2", "5,4", "#9a998f") if off(n) else ("#2a78d6", "", "#0b0b0b")
        if special:
            stroke, txt = "#52514e", "#52514e"
        parts.append(f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="{22 if special else 8}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" stroke-dasharray="{dash}"/>')
        for i, line in enumerate(lab):
            ty = y + H / 2 + (i - (len(lab) - 1) / 2) * 14 + 4
            parts.append(f'<text x="{x + W / 2}" y="{ty}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{txt}">{html.escape(line)}</text>')
        if off(n) and not special:
            parts.append(f'<text x="{x + W - 6}" y="{y + 12}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#9a998f">bypassed</text>')
    parts.append("</svg>")
    return "".join(parts)


def spec_flow_html(toggles: dict) -> str:
    chips = []
    for i, (label, key) in enumerate(SPEC_FLOW):
        on = True if key is None else bool(toggles.get(key, True))
        style = ("background:#e8f0fb;border:1.5px solid #2a78d6;color:#0b0b0b" if on else
                 "background:#f6f6f4;border:1.5px dashed #b9b8b2;color:#9a998f")
        chips.append(f'<span style="{style};padding:6px 12px;border-radius:16px;font:13px sans-serif">{html.escape(label)}{"" if on else " (off)"}</span>')
        if i < len(SPEC_FLOW) - 1:
            chips.append('<span style="color:#52514e">&rarr;</span>')
    return '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0">' + "".join(chips) + "</div>"
