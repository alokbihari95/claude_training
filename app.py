"""PwC India Advisory - Standard RAG vs Multi-HyDE RAG vs Graph RAG (LangChain + LangGraph). Run: python app.py

Single-user local app: tab-to-tab settings (practice area, hyperparameters, workflow toggles, graph selection) live in
one module-level dict, so changes in tabs 2-4 really change what tab 1 retrieves.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

import gradio as gr

import config
from src import evaluate as ev
from src.engines import run_engine
from src.retrieval import get_index, reset_index
from ui import charts, graph_canvas, workflow_canvas

LOCK = threading.Lock()
S: dict = {
    "hp": config.DEFAULTS.as_dict(),
    "toggles": dict(workflow_canvas.DEFAULT_TOGGLES),
    "kg_nodes": set(), "kg_edges": set(),      # exclusions clicked on the 3D canvas
    "dd_nodes": set(),                         # exclusions chosen from the dropdown fallback
    "last": None,                              # last chat run (live telemetry)
    "index_chunk": (config.DEFAULTS.chunk_size, config.DEFAULTS.chunk_overlap),
}
CITE = re.compile(r"\[((?:TAX|GRC|DEAL)-\d+[^\]]*)\]")


def excluded_nodes() -> list[str]:
    return sorted(S["kg_nodes"] | S["dd_nodes"])


def gold_questions() -> list[dict]:
    return ev.load_gold()


# ---------------------------------------------------------------- tab 1
def ingest_uploads(files, area):
    if not files:
        yield "Drop PDF/DOCX files first."
        return
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = []
    for f in files:
        p = Path(f if isinstance(f, str) else f.name)
        if p.suffix.lower() not in {".pdf", ".docx"}:
            yield f"Skipping {p.name}: only PDF/DOCX are accepted here."
            continue
        d = config.UPLOAD_DIR / p.name
        shutil.copy(p, d)
        dest.append(d)
    if not dest:
        return
    from src.ingest import ingest_paths

    lines: list[str] = []
    hp = S["hp"]
    done = {}

    def worker():
        try:
            done["r"] = ingest_paths(dest, area if area != config.ALL_AREAS else None, hp["chunk_size"], hp["chunk_overlap"], True, lines.append)
        except Exception as ex:  # noqa: BLE001
            done["e"] = ex

    t = threading.Thread(target=worker)
    t.start()
    while t.is_alive():
        t.join(timeout=1.0)
        yield "\n".join(lines)
    yield "\n".join(lines) + (f"\nFAILED: {done['e']!r}" if "e" in done else "\nReady: ask questions about the new documents.")
    reset_index()


def render_answer(text: str) -> str:
    return CITE.sub(lambda m: f"`[{m.group(1)}]`", text)


def sources_md(res) -> str:
    if not res.contexts:
        return "_No passages retrieved._"
    out = []
    for i, c in enumerate(res.contexts, 1):
        m = c["metadata"]
        pg = f", page {m['page']}" if m.get("page") else ""
        out.append(f"**[{i}] {m.get('doc_id')}**: {m.get('title')}{pg} · via `{c.get('via') or '-'}` · score {c.get('rerank_score', c.get('score', 0)):.3f}")
        out.append(f"> {c['text'][:260].replace(chr(10), ' ')}...")
        for t in c.get("triples", [])[:4]:
            out.append(f"- graph: `{t}`")
    return "\n\n".join(out)


def telemetry_md(res) -> str:
    t, u = res.timings, res.usage
    hops = max([x.get("hop", 0) for x in res.trace] + [0])
    steps = " → ".join(x["node"] + (f"[{x['hop']}]" if "hop" in x and x["node"] == "expand_subgraph" else "") for x in res.trace)
    return (f"**{config.ENGINE_LABELS[res.engine]}** · total **{res.latency_ms:,.0f} ms** "
            f"(retrieval {t.get('retrieval', 0):,.0f} · graph {t.get('graph', 0):,.0f} · LLM {t.get('llm', 0):,.0f}) · "
            f"{u.get('llm_calls', 0)} LLM calls · {u.get('input_tokens', 0):,} in / {u.get('output_tokens', 0):,} out tokens"
            + (f" · hops {hops}" if res.engine == "graph" else "") + f"\n\n`{steps}`")


def chat(message, history, engine, area):
    if not message or not message.strip():
        return history, "", "", ""
    idx = get_index()
    history = list(history or [])
    if not idx.ready:
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": "The index is empty. Run `python -m src.ingest` or upload documents first."}]
        return history, "", "", ""
    try:
        res = run_engine(engine, message, area, S["hp"], S["toggles"], excluded_nodes(), sorted(S["kg_edges"]))
    except Exception as ex:  # noqa: BLE001
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": f"Error: `{ex!r}`"}]
        return history, "", "", ""
    S["last"] = res
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": render_answer(res.answer)}]
    return history, "", sources_md(res), telemetry_md(res)


# ---------------------------------------------------------------- tab 2
def graph_choices():
    G = get_index().graph.G
    ns = sorted(((n, d) for n, d in G.nodes(data=True) if d.get("type", "").lower() not in {"value", "date"}), key=lambda x: -G.degree(x[0]))[:120]
    return [(f"{d.get('name', n)} ({G.degree(n)})", n) for n, d in ns]


def draw_graph(areas, literals, min_deg):
    G = get_index().graph.G
    if G.number_of_nodes() == 0:
        return "<p>No graph yet. Run ingestion first.</p>", "empty"
    fig, n, e = graph_canvas.build_figure(G, set(areas or []), literals, int(min_deg), S["kg_nodes"] | S["dd_nodes"], S["kg_edges"])
    return graph_canvas.iframe_html(fig), f"{n} entities · {e} relations shown · {len(S['kg_nodes'] | S['dd_nodes'])} nodes and {len(S['kg_edges'])} edges excluded from retrieval"


def on_graph_click(evt: gr.EventData):
    try:
        S["kg_nodes"] = set(json.loads(getattr(evt, "excluded_nodes", "[]") or "[]"))
        S["kg_edges"] = set(json.loads(getattr(evt, "excluded_edges", "[]") or "[]"))
    except Exception:  # noqa: BLE001
        pass
    return f"Selection updated: {len(S['kg_nodes'] | S['dd_nodes'])} nodes and {len(S['kg_edges'])} edges excluded from Graph RAG retrieval."


def set_dd(nodes):
    S["dd_nodes"] = set(nodes or [])
    return f"{len(S['kg_nodes'] | S['dd_nodes'])} nodes and {len(S['kg_edges'])} edges excluded from retrieval."


def clear_selection(areas, literals, min_deg):
    S["kg_nodes"], S["kg_edges"], S["dd_nodes"] = set(), set(), set()
    h, st = draw_graph(areas, literals, min_deg)
    return h, st, []


# ---------------------------------------------------------------- tab 3
LABEL_TO_KEY = {v[0]: k for k, v in workflow_canvas.TOGGLES.items()}


def apply_toggles(selected):
    S["toggles"] = {k: (workflow_canvas.TOGGLES[k][0] in (selected or [])) for k in workflow_canvas.TOGGLES}
    return workflow_canvas.workflow_svg(S["toggles"]), workflow_canvas.spec_flow_html(S["toggles"])


def run_variant(question, area):
    if not question.strip():
        return "Enter a question.", "", ""
    res = run_engine("graph", question, area, S["hp"], S["toggles"], excluded_nodes(), sorted(S["kg_edges"]))
    S["last"] = res
    docs = ", ".join(dict.fromkeys(c["metadata"]["doc_id"] for c in res.contexts))
    return telemetry_md(res), f"Retrieved documents: **{docs}**", render_answer(res.answer)


# ---------------------------------------------------------------- tab 4
def apply_hp(chunk, overlap, hyde_n, hops, rrf, topk, temp):
    S["hp"] = {"chunk_size": int(chunk), "chunk_overlap": int(overlap), "hyde_n": int(hyde_n), "hop_radius": int(hops),
               "rrf_constant": int(rrf), "top_k": int(topk), "temperature": float(temp)}
    stale = (int(chunk), int(overlap)) != S["index_chunk"]
    msg = "Applied. Hop radius, Multi-HyDE N, RRF constant, top-k and temperature take effect on the next query."
    if stale:
        msg += f"\n\n**Index is stale**: it was built at {S['index_chunk'][0]}/{S['index_chunk'][1]} tokens; press *Re-index* to rebuild at {int(chunk)}/{int(overlap)}."
    return msg


def reindex():
    from src.ingest import build_all

    log: list[str] = []
    build_all(rebuild=True, chunk_size=S["hp"]["chunk_size"], chunk_overlap=S["hp"]["chunk_overlap"], log=log.append)
    S["index_chunk"] = (S["hp"]["chunk_size"], S["hp"]["chunk_overlap"])
    reset_index()
    return "Re-indexed.\n\n" + "\n".join(log[-6:])


def kpi_html(engine):
    res = ev.load_results()
    tiles = []
    if res and engine in res["aggregate"]["overall"]:
        a = res["aggregate"]["overall"][engine]
        er = res["er_resolution"]
        f = lambda x, p=2: "n/a" if x is None else f"{x:.{p}f}"  # noqa: E731
        tiles = [charts.kpi_tile("Retrieval Context Recall", f(a["rcr"]), f"{a['n']} questions · benchmark {res['created']}"),
                 charts.kpi_tile("Faithfulness (1 - hallucination)", f(a["faithfulness"]), f"hallucination {f(a['hallucination'])}"),
                 charts.kpi_tile("Entity-Relation Resolution", f"{er['group_accuracy']:.2f}" if engine == "graph" else "n/a",
                                 f"{er['groups_correct']}/{er['groups_evaluated']} entity groups · links {er['link_recall']:.2f}" if engine == "graph" else "graph index only"),
                 charts.kpi_tile("End-to-end latency", f"{a['latency_ms']:,.0f} ms", "retrieval {:.0f} · graph {:.0f} · LLM {:.0f}".format(
                     *(a["latency_breakdown_ms"].get(k) or 0 for k in ("retrieval", "graph", "llm"))))]
    else:
        tiles = [charts.kpi_tile(k, "n/a", "run the benchmark in tab 5") for k in ("Retrieval Context Recall", "Faithfulness", "Entity-Relation Resolution", "End-to-end latency")]
    live = S["last"]
    live_html = "<p style='color:#52514e'>Ask a question in tab 1 to see live latency telemetry.</p>" if not live else \
        charts.kpi_row([charts.kpi_tile("Last query latency", f"{live.latency_ms:,.0f} ms", config.ENGINE_LABELS[live.engine]),
                        charts.kpi_tile("Retrieval / Graph / LLM", " / ".join(f"{live.timings.get(k, 0):,.0f}" for k in ("retrieval", "graph", "llm")), "ms"),
                        charts.kpi_tile("Tokens in / out", f"{live.usage.get('input_tokens', 0):,} / {live.usage.get('output_tokens', 0):,}", f"{live.usage.get('llm_calls', 0)} LLM calls")])
    return ("<h4 style='margin:6px 0'>Benchmark KPIs: " + config.ENGINE_LABELS[engine] + "</h4>" + charts.kpi_row(tiles)
            + "<h4 style='margin:14px 0 6px'>Live telemetry (last query)</h4>" + live_html)


# ---------------------------------------------------------------- tab 5
def _cell(agg, t, e, key="rcr"):
    c = agg["by_type"][t].get(e, {})
    return "n/a" if c.get("n", 0) == 0 or c.get(key) is None else f"{c[key]:.2f} (n={c['n']})"


def matrix_rows(res: dict):
    agg, cmp_ = res["aggregate"], res["spec_comparison"]
    spec = json.loads(config.SPEC_EXPECTATIONS_PATH.read_text())["matrix"]
    flag = lambda ok: "✅ matches" if ok else "❌ differs"  # noqa: E731
    L = config.ENGINE_LABELS
    nm = lambda order: " > ".join(L[e].split(" ")[0] if e != "graph" else "Graph" for e in order)  # noqa: E731
    rows = []
    for t, name in (("simple", "Simple keyword queries (RCR)"), ("multihop", "Multi-hop relational (RCR)"), ("vocab_mismatch", "Domain vocabulary mismatch (RCR)")):
        rows.append([name] + [_cell(agg, t, e) for e in config.ENGINES] + [nm(cmp_[t]["expected_order_best_first"]) + " (best first)", flag(cmp_[t]["match"])])
    lat = agg["overall"]
    rows.append(["Average latency (ms)"] + [f"{lat[e]['latency_ms']:,.0f}" if lat[e]["latency_ms"] else "n/a" for e in config.ENGINES]
                + [f"{nm(cmp_['latency']['expected_order_fastest_first'])} (fastest first; spec ~450/850/1,200 ms)", flag(cmp_["latency"]["match"])])
    rows.append(["Hallucination risk (1 − faithfulness)"] + [f"{lat[e]['hallucination']:.2f}" if lat[e]["hallucination"] is not None else "n/a" for e in config.ENGINES]
                + ["Graph lowest", flag(cmp_["hallucination"]["match"])])
    rows.append(["Best enterprise use case (spec, not measured)"] + [spec["best_use_case"][e] for e in config.ENGINES] + ["-", "-"])
    for key, label in (("faithfulness", "Faithfulness"), ("correctness", "Answer correctness (judge)"), ("doc_recall", "Gold-document recall")):
        rows.append([label] + [f"{lat[e][key]:.2f}" if lat[e][key] is not None else "n/a" for e in config.ENGINES] + ["-", "-"])
    er = res["er_resolution"]
    rows.append(["Entity-Relation Resolution accuracy"] + ["n/a", "n/a", f"{er['group_accuracy']:.2f} ({er['groups_correct']}/{er['groups_evaluated']}); links {er['link_recall']:.2f}"] + ["graph index only", "-"])
    return rows


MATRIX_HEAD = ["Metric"] + [config.ENGINE_LABELS[e] for e in config.ENGINES] + ["Spec expectation", "Measured vs spec"]


def show_results(res):
    if not res:
        empty = [["Run the benchmark to populate this matrix"] + [""] * 5]
        return empty, None, None, None, None, "No benchmark results yet."
    agg = res["aggregate"]
    note = (f"**{res['n_questions']} questions** × 3 engines · generation `{res['config']['llm']}` · judge `{res['config']['judge']}` · {res['created']}. "
            f"Failed runs: {agg['failed']}. Spec columns are *expectations*; the ✅/❌ column compares measured ordering to them (ties within 0.02 allowed).")
    return (matrix_rows(res), charts.rcr_by_type(agg), charts.metric_bars(agg, "faithfulness", "Faithfulness (supported claims / total)", "0-1", yr=[0, 1.15]),
            charts.latency_stack(agg), charts.metric_bars(agg, "correctness", "Answer correctness (LLM judge)", "0-1", yr=[0, 1.15]), note)


def run_live(engines, limit, progress=gr.Progress()):
    engines = [k for k, v in config.ENGINE_LABELS.items() if v in (engines or [])]
    if not engines:
        raise gr.Error("Pick at least one engine.")
    res = ev.run_benchmark(engines=engines, limit=int(limit) or None, hp=S["hp"], toggles=S["toggles"], excluded_nodes=excluded_nodes(),
                           excluded_edges=sorted(S["kg_edges"]), progress=lambda f, m: progress(f, desc=m))
    return show_results(res)


# ---------------------------------------------------------------- UI
def build() -> gr.Blocks:
    areas = [c for c in config.AREA_CHOICES]
    with gr.Blocks(title="PwC India Advisory: RAG Benchmark") as demo:
        gr.Markdown("# PwC India Advisory: Standard vs Multi-HyDE vs Graph RAG\n"
                    "LangChain + LangGraph benchmark on a **synthetic** Tax / Risk Advisory / Deal corpus. "
                    "All regulatory text, section numbers and figures are fictional.")
        with gr.Tabs():
            # ---- Tab 1
            with gr.Tab("1 · Ingestion & Chat Console"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        files = gr.File(label="Drag & drop PDF / DOCX", file_count="multiple", file_types=[".pdf", ".docx"])
                        area_up = gr.Dropdown(areas, value="tax", label="Practice area for uploaded files")
                        ingest_btn = gr.Button("Ingest (6-step pipeline)")
                        ingest_log = gr.Textbox(label="Ingestion log", lines=7, interactive=False)
                        area = gr.Dropdown(areas, value=config.ALL_AREAS, label="Practice Area (guardrails + retrieval filter)")
                        engine = gr.Radio([(v, k) for k, v in config.ENGINE_LABELS.items()], value="graph", label="RAG Engine")
                        gold = gr.Dropdown([(f"{q['id']} · {q['question'][:70]}", q["question"]) for q in gold_questions()], label="Or load a benchmark question")
                    with gr.Column(scale=2):
                        bot = gr.Chatbot(height=430, label="Advisory assistant", buttons=["copy"])
                        msg = gr.Textbox(placeholder="Ask a regulatory question...", show_label=False)
                        send = gr.Button("Send", variant="primary")
                        tele = gr.Markdown()
                        with gr.Accordion("Retrieved sources and graph facts", open=False):
                            src = gr.Markdown()
                ingest_btn.click(ingest_uploads, [files, area_up], ingest_log)
                gold.change(lambda q: q, gold, msg)
                for trig in (send.click, msg.submit):
                    trig(chat, [msg, bot, engine, area], [bot, msg, src, tele])

            # ---- Tab 2
            with gr.Tab("2 · Knowledge Graph Canvas") as tab2:
                with gr.Row():
                    g_areas = gr.CheckboxGroup([(v["label"], k) for k, v in config.PRACTICE_AREAS.items()], value=list(config.PRACTICE_AREAS), label="Practice areas shown")
                    g_lit = gr.Checkbox(False, label="Show literal values (rates, dates, amounts)")
                    g_deg = gr.Slider(0, 6, value=1, step=1, label="Min degree")
                with gr.Row():
                    g_refresh = gr.Button("Redraw")
                    g_clear = gr.Button("Clear selection")
                g_status = gr.Markdown()
                g_html = gr.HTML(js_on_load=graph_canvas.BRIDGE_JS)
                g_dd = gr.Dropdown(graph_choices() if get_index().graph.G.number_of_nodes() else [], multiselect=True, label="Fallback: exclude entities from a list (merged with canvas clicks)")
                g_note = gr.Markdown("Excluded nodes/edges (grey) are removed from **Graph RAG** traversal and its graph rank list; Standard and HyDE do not use the graph.")
                g_refresh.click(draw_graph, [g_areas, g_lit, g_deg], [g_html, g_status])
                tab2.select(draw_graph, [g_areas, g_lit, g_deg], [g_html, g_status])
                g_html.click(on_graph_click, None, g_note)
                g_dd.change(set_dd, g_dd, g_note)
                g_clear.click(clear_selection, [g_areas, g_lit, g_deg], [g_html, g_status, g_dd])

            # ---- Tab 3
            with gr.Tab("3 · LangGraph Workflow Canvas"):
                gr.Markdown("Execution DAG of the **Graph RAG** LangGraph (drawn from the compiled graph). Untick a node to bypass it and test pipeline variants.")
                labels = [v[0] for v in workflow_canvas.TOGGLES.values()]
                tg = gr.CheckboxGroup(labels, value=[workflow_canvas.TOGGLES[k][0] for k, on in workflow_canvas.DEFAULT_TOGGLES.items() if on], label="Active workflow nodes / edges")
                spec_flow = gr.HTML(workflow_canvas.spec_flow_html(S["toggles"]))
                svg = gr.HTML(workflow_canvas.workflow_svg(S["toggles"]))
                tg.change(apply_toggles, tg, [svg, spec_flow])
                gr.Markdown("### Test this variant")
                with gr.Row():
                    v_q = gr.Textbox(label="Question", scale=3, value="Is the unpaid withholding exposure on Meridian's management fees subject to the SPA indemnity cap?")
                    v_area = gr.Dropdown(areas, value="deals", label="Practice area", scale=1)
                    v_go = gr.Button("Run Graph RAG with these toggles", scale=1)
                v_tele, v_docs, v_ans = gr.Markdown(), gr.Markdown(), gr.Markdown()
                v_go.click(run_variant, [v_q, v_area], [v_tele, v_docs, v_ans])

            # ---- Tab 4
            with gr.Tab("4 · Hyperparameters & KPIs") as tab4:
                d = config.DEFAULTS
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### The 4 configurable hyperparameters")
                        h_chunk = gr.Slider(200, 1200, value=d.chunk_size, step=50, label="1 · Chunk size (tokens)")
                        h_over = gr.Slider(0, 200, value=d.chunk_overlap, step=10, label="1 · Chunk overlap (tokens)")
                        h_n = gr.Slider(1, 5, value=d.hyde_n, step=1, label="2 · Multi-HyDE passage count N")
                        h_k = gr.Slider(1, 4, value=d.hop_radius, step=1, label="3 · Graph traversal hop radius k")
                        h_rrf = gr.Slider(1, 120, value=d.rrf_constant, step=1, label="4 · Fusion RRF constant (BM25 + vector + graph)")
                        with gr.Accordion("Other (fixed across engines)", open=False):
                            h_top = gr.Slider(1, 10, value=d.top_k, step=1, label="top-k contexts")
                            h_temp = gr.Slider(0, 1, value=d.temperature, step=0.1, label="Generation temperature")
                        with gr.Row():
                            h_apply = gr.Button("Apply", variant="primary")
                            h_reidx = gr.Button("Re-index (after chunk change)")
                        h_msg = gr.Markdown()
                    with gr.Column():
                        k_engine = gr.Radio([(v, k) for k, v in config.ENGINE_LABELS.items()], value="graph", label="Engine")
                        k_refresh = gr.Button("Refresh KPIs")
                        k_html = gr.HTML()
                hp_inputs = [h_chunk, h_over, h_n, h_k, h_rrf, h_top, h_temp]
                h_apply.click(apply_hp, hp_inputs, h_msg)
                h_reidx.click(reindex, None, h_msg)
                for t in (k_refresh.click, k_engine.change):
                    t(kpi_html, k_engine, k_html)
                tab4.select(kpi_html, k_engine, k_html)

            # ---- Tab 5
            with gr.Tab("5 · Comparative Benchmark") as tab5:
                with gr.Row():
                    b_eng = gr.CheckboxGroup(list(config.ENGINE_LABELS.values()), value=list(config.ENGINE_LABELS.values()), label="Engines")
                    b_lim = gr.Slider(0, 24, value=0, step=1, label="Limit questions (0 = all 24)")
                    b_go = gr.Button("Run live benchmark", variant="primary")
                b_note = gr.Markdown()
                b_tab = gr.Dataframe(headers=MATRIX_HEAD, interactive=False, wrap=True, label="Comparative matrix (table view)")
                with gr.Row():
                    p1, p2 = gr.Plot(label="RCR by question type"), gr.Plot(label="Faithfulness")
                with gr.Row():
                    p3, p4 = gr.Plot(label="Latency breakdown"), gr.Plot(label="Answer correctness")
                outs = [b_tab, p1, p2, p3, p4, b_note]
                b_go.click(run_live, [b_eng, b_lim], outs)
                tab5.select(lambda: show_results(ev.load_results()), None, outs)
            demo.load(lambda: show_results(ev.load_results()), None, outs)
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(__import__("os").environ.get("PORT", 7870)), theme=gr.themes.Soft(primary_hue="blue"))
