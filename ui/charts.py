"""Benchmark charts and KPI tiles. Colours: validated categorical slots 1-3 (blue / orange / aqua) in fixed engine order.
Aqua is below 3:1 on the light surface, so every bar carries a visible value label and a table view sits beside the chart."""
from __future__ import annotations

import html

import plotly.graph_objects as go

import config

ENGINE_COLOR = {"standard": "#2a78d6", "hyde": "#eb6834", "graph": "#1baf7a"}
TYPE_LABEL = {"simple": "Simple keyword", "multihop": "Multi-hop relational", "vocab_mismatch": "Vocabulary mismatch"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#ecebe6", "#fcfcfb"


def _base(fig: go.Figure, title: str, ytitle: str, height: int = 340) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=INK), x=0.0), height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      margin=dict(l=50, r=10, t=50, b=40), font=dict(color=MUTED), barmode="group", bargap=0.35, bargroupgap=0.08,
                      legend=dict(orientation="h", y=1.14, x=0.0))
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def rcr_by_type(agg: dict) -> go.Figure:
    fig = go.Figure()
    for e in config.ENGINES:
        ys = [agg["by_type"][t].get(e, {}).get("rcr") for t in config.QUESTION_TYPES]
        ns = [agg["by_type"][t].get(e, {}).get("n", 0) for t in config.QUESTION_TYPES]
        fig.add_bar(x=[TYPE_LABEL[t] for t in config.QUESTION_TYPES], y=ys, name=config.ENGINE_LABELS[e], marker=dict(color=ENGINE_COLOR[e], cornerradius=4),
                    text=[f"{y:.2f}" if y is not None else "n/a" for y in ys], textposition="outside", customdata=ns,
                    hovertemplate="%{x}<br>" + config.ENGINE_LABELS[e] + ": %{y:.2f} (n=%{customdata})<extra></extra>")
    _base(fig, "Retrieval Context Recall by question type", "RCR (0-1)").update_yaxes(range=[0, 1.15])
    return fig


def metric_bars(agg: dict, key: str, title: str, ytitle: str, fmt: str = "{:.2f}", yr: list | None = None) -> go.Figure:
    fig = go.Figure()
    vals = [agg["overall"].get(e, {}).get(key) for e in config.ENGINES]
    fig.add_bar(x=[config.ENGINE_LABELS[e] for e in config.ENGINES], y=vals, marker=dict(color=[ENGINE_COLOR[e] for e in config.ENGINES], cornerradius=4),
                text=[fmt.format(v) if v is not None else "n/a" for v in vals], textposition="outside", showlegend=False,
                hovertemplate="%{x}: %{y}<extra></extra>")
    _base(fig, title, ytitle, 320)
    if yr:
        fig.update_yaxes(range=yr)
    return fig


def latency_stack(agg: dict) -> go.Figure:
    """Stacked retrieval / graph traversal / LLM inference per engine (segments are parts of one total, so a stack is legitimate)."""
    fig = go.Figure()
    shades = {"retrieval": "#2a78d6", "graph": "#eda100", "llm": "#6250d6"}   # stage identity, not engine identity
    names = {"retrieval": "Retrieval", "graph": "Graph traversal", "llm": "LLM inference"}
    for stage in ("retrieval", "graph", "llm"):
        ys = [(agg["overall"].get(e, {}).get("latency_breakdown_ms") or {}).get(stage) or 0 for e in config.ENGINES]
        fig.add_bar(x=[config.ENGINE_LABELS[e] for e in config.ENGINES], y=ys, name=names[stage], marker=dict(color=shades[stage]),
                    text=[f"{y:,.0f}" if y >= 150 else "" for y in ys], textposition="inside", textfont=dict(color="#ffffff"),
                    hovertemplate=names[stage] + ": %{y:,.0f} ms<extra></extra>")
    _base(fig, "End-to-end latency breakdown (ms, mean per query)", "ms", 340).update_layout(barmode="stack", bargap=0.45)
    return fig


def kpi_tile(label: str, value: str, sub: str = "") -> str:
    return (f'<div style="flex:1;min-width:170px;border:1px solid #e1e0da;border-radius:10px;padding:12px 14px;background:{SURFACE}">'
            f'<div style="font:12px sans-serif;color:{MUTED}">{html.escape(label)}</div>'
            f'<div style="font:600 28px sans-serif;color:{INK};margin:2px 0">{html.escape(value)}</div>'
            f'<div style="font:12px sans-serif;color:{MUTED}">{html.escape(sub)}</div></div>')


def kpi_row(tiles: list[str]) -> str:
    return '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(tiles) + "</div>"
