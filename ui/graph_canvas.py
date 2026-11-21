"""Tab 2: interactive spherical 3D knowledge-graph canvas.

The figure lives in an <iframe srcdoc> (plotly.js inlined, no CDN needed). Clicking a node or an edge midpoint
toggles it excluded/included; the iframe posts the selection to the parent page, where gr.HTML's js_on_load
forwards it to Python with trigger(). Python stores it and the engines use it as a retrieval filter.
"""
from __future__ import annotations

import html
import json
import math
from functools import lru_cache

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from src.graph_store import LITERAL_TYPES

AREA_COLORS = {"tax": "#2a78d6", "grc": "#eb6834", "deals": "#1baf7a"}   # validated categorical slots 1-3
AREA_LABEL = {"tax": "Tax & Regulatory", "grc": "Risk Advisory & GRC", "deals": "Deal Advisory"}
EXCLUDED = "#b9b8b2"
BRIDGE_JS = """
window.addEventListener('message', (ev) => {
  const d = ev.data;
  if (d && d.kg_selection) { trigger('click', {excluded_nodes: JSON.stringify(d.kg_selection.nodes), excluded_edges: JSON.stringify(d.kg_selection.edges)}); }
});
"""


def _fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    pts, golden = [], math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - y * y))
        t = golden * i
        pts.append((math.cos(t) * r, y, math.sin(t) * r))
    return pts


def _primary_area(d: dict) -> str:
    areas = d.get("areas") or ["tax"]
    return areas[0] if len(areas) == 1 else "multi"


def build_figure(G, areas: set[str], show_literals: bool, min_degree: int, excl_nodes: set[str], excl_edges: set[str]) -> tuple[go.Figure, int, int]:
    keep = [n for n, d in G.nodes(data=True)
            if (show_literals or d.get("type", "").lower() not in LITERAL_TYPES)
            and (not areas or set(d.get("areas", [])) & areas) and G.degree(n) >= min_degree]
    # sort so each practice area occupies a latitude band of the sphere (Fibonacci order runs pole to pole)
    order = {"tax": 0, "multi": 1, "deals": 2, "grc": 3}
    keep.sort(key=lambda n: (order.get(_primary_area(G.nodes[n]), 1), -G.degree(n)))
    pos = dict(zip(keep, _fibonacci_sphere(len(keep))))
    ks = set(keep)

    fig = go.Figure()
    xs, ys, zs, colors, sizes, texts, ids = [], [], [], [], [], [], []
    for n in keep:
        d = G.nodes[n]
        x, y, z = pos[n]
        xs.append(x); ys.append(y); zs.append(z)
        colors.append(EXCLUDED if n in excl_nodes else AREA_COLORS.get(_primary_area(d), "#6b6a66"))
        sizes.append(5 + min(G.degree(n), 14))
        texts.append(f"<b>{html.escape(d.get('name', n))}</b><br>{d.get('type', '')}<br>docs: {', '.join(d.get('docs', []))}<br>degree {G.degree(n)}")
        ids.append(n)
    ex, ey, ez, mx, my, mz, mtext, mids, mcolors = [], [], [], [], [], [], [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        if u in ks and v in ks:
            (x0, y0, z0), (x1, y1, z1) = pos[u], pos[v]
            gone = k in excl_edges
            if not gone:
                ex += [x0, x1, None]; ey += [y0, y1, None]; ez += [z0, z1, None]
            mx.append((x0 + x1) / 2); my.append((y0 + y1) / 2); mz.append((z0 + z1) / 2)
            mtext.append(f"{html.escape(G.nodes[u].get('name', u))} <b>-[{html.escape(d['relation'])}]-></b> {html.escape(G.nodes[v].get('name', v))}<br>{d['doc_id']}")
            mids.append(k)
            mcolors.append(EXCLUDED if gone else "#52514e")
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="rgba(82,81,78,0.35)", width=1.5), hoverinfo="skip", name="relations", showlegend=False))
    fig.add_trace(go.Scatter3d(x=mx, y=my, z=mz, mode="markers", marker=dict(size=3, color=mcolors), text=mtext, hoverinfo="text",
                               customdata=mids, name="edge (click to toggle)", showlegend=False))
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=sizes, color=colors, line=dict(color="#fcfcfb", width=1.5)),
                               text=texts, hoverinfo="text", customdata=ids, name="entities", showlegend=False))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color="#6b6a66"), name="Spans several areas"))
    for a, c in AREA_COLORS.items():   # legend proxies so identity is never colour-alone
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color=c), name=AREA_LABEL[a]))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#fcfcfb", height=640,
                      legend=dict(orientation="h", y=0.02, x=0.0, font=dict(color="#52514e")),
                      scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="cube",
                                 camera=dict(eye=dict(x=1.5, y=0.9, z=0.9))))
    return fig, len(keep), len(mids)


@lru_cache(maxsize=1)
def _plotly_js() -> str:
    return get_plotlyjs()


def iframe_html(fig: go.Figure) -> str:
    """Self-contained page: figure + click toggling + postMessage to the parent."""
    fig_json = fig.to_json()
    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fcfcfb;font-family:sans-serif}}
#hint{{position:absolute;top:6px;left:10px;font-size:12px;color:#52514e;z-index:5;background:#fcfcfbcc;padding:2px 6px;border-radius:4px}}</style>
<script>{_plotly_js()}</script></head><body><div id="hint">Click a node or a small edge marker to exclude/include it in retrieval. Drag to rotate.</div>
<div id="g"></div><script>
const fig = {fig_json};
const EXC = '{EXCLUDED}';
Plotly.newPlot('g', fig.data, fig.layout, {{responsive:true, displaylogo:false}}).then(gd => {{
  const base = {{}};   // original colours per trace so a second click restores them
  [1,2].forEach(t => base[t] = Array.isArray(gd.data[t].marker.color) ? gd.data[t].marker.color.slice() : []);
  const nodes = new Set(), edges = new Set();
  gd.data[2].marker.color.forEach((c,i) => {{ if (c === EXC) nodes.add(gd.data[2].customdata[i]); }});
  gd.data[1].marker.color.forEach((c,i) => {{ if (c === EXC) edges.add(gd.data[1].customdata[i]); }});
  const orig2 = gd.data[2].customdata.map((id,i) => base[2][i] === EXC ? '#6b6a66' : base[2][i]);
  const orig1 = gd.data[1].customdata.map((id,i) => base[1][i] === EXC ? '#52514e' : base[1][i]);
  gd.on('plotly_click', ev => {{
    const p = ev.points[0]; if (!p) return;
    const set = p.curveNumber === 2 ? nodes : (p.curveNumber === 1 ? edges : null); if (!set) return;
    const id = p.customdata; set.has(id) ? set.delete(id) : set.add(id);
    const t = p.curveNumber, arr = (t === 2 ? gd.data[2].customdata : gd.data[1].customdata);
    const cols = arr.map((x,i) => set.has(x) ? EXC : (t === 2 ? orig2[i] : orig1[i]));
    Plotly.restyle(gd, {{'marker.color': [cols]}}, [t]);
    parent.postMessage({{kg_selection: {{nodes: [...nodes], edges: [...edges]}}}}, '*');
  }});
}});
</script></body></html>"""
    return f'<iframe srcdoc="{html.escape(page, quote=True)}" style="width:100%;height:650px;border:1px solid #e1e0da;border-radius:8px" title="Knowledge graph"></iframe>'
