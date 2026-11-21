"""Benchmark runner: all engines x gold questions -> KPIs -> results/benchmark.json (+ .csv).

Pipelines run SEQUENTIALLY so latency is not inflated by contention; judging runs in parallel afterwards.

CLI: python -m src.evaluate [--limit N] [--engines standard hyde graph] [--types simple multihop vocab_mismatch]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import config
from src import metrics
from src.engines import run_engine
from src.llm import UsageTracker, get_llm

RESULTS_PATH = config.RESULTS_DIR / "benchmark.json"
TIE_TOL = 0.02


def load_gold() -> list[dict]:
    return json.loads(config.GOLD_QA_PATH.read_text())


def run_benchmark(engines: list[str] | None = None, types: list[str] | None = None, limit: int | None = None,
                  hp: dict | None = None, toggles: dict | None = None, excluded_nodes: list[str] | None = None,
                  excluded_edges: list[str] | None = None, progress: Callable[[float, str], None] | None = None,
                  save: bool = True) -> dict:
    engines = engines or config.ENGINES
    gold = [q for q in load_gold() if not types or q["type"] in types]
    if limit:
        gold = gold[:limit]
    total = len(gold) * len(engines)
    records, done = [], 0
    if progress:
        progress(0.0, "warming up models")
    from src.llm import warmup
    warmup()
    run_engine(engines[0], "warm up question", "all", hp, toggles)   # also warms the index and graph
    for q in gold:
        for e in engines:
            if progress:
                progress(done / total, f"{e} · {q['id']}")
            try:
                r = run_engine(e, q["question"], q["practice_area"], hp, toggles, excluded_nodes, excluded_edges)
                records.append({"qid": q["id"], "type": q["type"], "engine": e, "answer": r.answer, "contexts": r.contexts,
                                "latency_ms": r.latency_ms, "timings": r.timings, "usage": r.usage, "trace": r.trace, "error": None})
            except Exception as ex:  # noqa: BLE001  one failed run must not sink the whole benchmark
                records.append({"qid": q["id"], "type": q["type"], "engine": e, "answer": "", "contexts": [], "latency_ms": None,
                                "timings": {}, "usage": {}, "trace": [], "error": repr(ex)})
            done += 1
    if progress:
        progress(0.9, "judging")
    judge = get_llm(model=config.JUDGE_MODEL, temperature=0.0, max_tokens=6000)
    by_id = {q["id"]: q for q in gold}

    def judge_one(rec: dict) -> dict:
        if rec["error"]:
            return rec
        q = by_id[rec["qid"]]
        try:
            rec["rcr"] = metrics.judge_rcr(judge, q, rec["contexts"])
            rec.update(metrics.judge_answer(judge, q, rec["answer"], rec["contexts"]))
        except Exception as ex:  # noqa: BLE001
            rec["error"] = f"judge: {ex!r}"
            return rec
        rec["doc_recall"] = metrics.doc_recall(q, rec["contexts"])
        rec["has_citation"] = metrics.has_citation(rec["answer"])
        return rec

    with ThreadPoolExecutor(6) as ex:
        records = list(ex.map(judge_one, records))
    er = metrics.er_resolution_accuracy()
    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "config": {"llm": config.LLM_MODEL, "judge": config.JUDGE_MODEL,
           "hp": {**config.DEFAULTS.as_dict(), **(hp or {})}, "toggles": toggles or {}}, "n_questions": len(gold),
           "records": records, "er_resolution": er}
    out["aggregate"] = aggregate(records, engines)
    out["spec_comparison"] = compare_to_spec(out["aggregate"])
    if save:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(out, indent=1))
        write_csv(records)
    if progress:
        progress(1.0, "done")
    return out


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(records: list[dict], engines: list[str]) -> dict:
    ok = [r for r in records if not r.get("error")]
    agg: dict = {"by_type": {}, "overall": {}, "failed": {e: sum(1 for r in records if r["engine"] == e and r.get("error")) for e in engines}}
    for e in engines:
        rows = [r for r in ok if r["engine"] == e]
        stage = {s: _mean([r["timings"].get(s) for r in rows]) for s in ("retrieval", "graph", "llm")}
        agg["overall"][e] = {
            "n": len(rows), "rcr": _mean([r["rcr"] for r in rows]), "faithfulness": _mean([r["faithfulness"] for r in rows]),
            "hallucination": _mean([r["hallucination"] for r in rows]), "correctness": _mean([r["correctness"] for r in rows]),
            "doc_recall": _mean([r["doc_recall"] for r in rows]), "citation_rate": _mean([float(r["has_citation"]) for r in rows]),
            "latency_ms": _mean([r["latency_ms"] for r in rows]), "latency_breakdown_ms": stage,
            "tokens_in": _mean([r["usage"].get("input_tokens") for r in rows]), "tokens_out": _mean([r["usage"].get("output_tokens") for r in rows]),
            "llm_calls": _mean([r["usage"].get("llm_calls") for r in rows])}
    for t in config.QUESTION_TYPES:
        agg["by_type"][t] = {}
        for e in engines:
            rows = [r for r in ok if r["engine"] == e and r["type"] == t]
            agg["by_type"][t][e] = {"n": len(rows), "rcr": _mean([r["rcr"] for r in rows]), "faithfulness": _mean([r["faithfulness"] for r in rows]),
                                    "hallucination": _mean([r["hallucination"] for r in rows]), "correctness": _mean([r["correctness"] for r in rows]),
                                    "doc_recall": _mean([r["doc_recall"] for r in rows]), "latency_ms": _mean([r["latency_ms"] for r in rows])}
    return agg


def _ordered(vals: dict[str, float | None], desc: list[str] | None = None, asc: list[str] | None = None) -> bool:
    """True if engines satisfy the expected order within TIE_TOL (ties are allowed in either direction)."""
    order = desc or asc
    seq = [vals.get(e) for e in order]
    if any(v is None for v in seq):
        return False
    for a, b in zip(seq, seq[1:]):
        if (desc and a < b - TIE_TOL) or (asc and a > b + TIE_TOL):
            return False
    return True


def compare_to_spec(agg: dict) -> dict:
    """Measured ordering vs the spec's expectations. These are hypotheses: mismatches are reported, not hidden."""
    spec = json.loads(config.SPEC_EXPECTATIONS_PATH.read_text())
    ordn = spec["ordinal_expectations"]
    out: dict = {}
    for t in config.QUESTION_TYPES:
        exp = ordn[t]
        vals = {e: agg["by_type"][t][e]["rcr"] for e in agg["by_type"][t]}
        out[t] = {"metric": "rcr", "expected_order_best_first": exp["order_desc"], "measured": vals,
                  "measured_order_best_first": sorted(vals, key=lambda e: -(vals[e] or 0)), "match": _ordered(vals, desc=exp["order_desc"]),
                  "n": {e: agg["by_type"][t][e]["n"] for e in agg["by_type"][t]}}
    lat = {e: agg["overall"][e]["latency_ms"] for e in agg["overall"]}
    out["latency"] = {"metric": "latency_ms", "expected_order_fastest_first": ordn["latency"]["order_asc"], "measured": lat,
                      "measured_order_fastest_first": sorted(lat, key=lambda e: lat[e] or 1e18),
                      "match": all(lat.get(a) is not None and lat.get(b) is not None and lat[a] <= lat[b]
                                   for a, b in zip(ordn["latency"]["order_asc"], ordn["latency"]["order_asc"][1:]))}
    hal = {e: agg["overall"][e]["hallucination"] for e in agg["overall"]}
    lowest = min(hal, key=lambda e: hal[e] if hal[e] is not None else 9)
    expected_lowest = ordn["hallucination"]["order_asc"][0]
    out["hallucination"] = {"metric": "hallucination (1 - faithfulness)", "expected_lowest": expected_lowest, "measured": hal,
                            "measured_lowest": lowest,
                            # a tie with the expected engine within tolerance still counts as consistent with "lowest"
                            "match": hal.get(expected_lowest) is not None and hal[expected_lowest] <= hal[lowest] + TIE_TOL}
    return out


def write_csv(records: list[dict]) -> None:
    cols = ["qid", "type", "engine", "rcr", "faithfulness", "hallucination", "correctness", "doc_recall", "has_citation", "latency_ms", "error"]
    with open(config.RESULTS_DIR / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def load_results() -> dict | None:
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else None


def print_summary(res: dict) -> None:
    agg = res["aggregate"]
    print(f"\nn_questions={res['n_questions']}  llm={res['config']['llm']}  judge={res['config']['judge']}")
    print(f"{'engine':9s} {'n':>3s} {'RCR':>6s} {'faith':>6s} {'correct':>8s} {'docRec':>7s} {'lat ms':>8s}  retr/graph/llm ms")
    for e, a in agg["overall"].items():
        b = a["latency_breakdown_ms"]
        f = lambda x: "  n/a" if x is None else f"{x:6.2f}"  # noqa: E731
        print(f"{e:9s} {a['n']:3d} {f(a['rcr'])} {f(a['faithfulness'])} {f(a['correctness'])}  {f(a['doc_recall'])} {a['latency_ms'] or 0:8.0f}  "
              f"{(b['retrieval'] or 0):.0f}/{(b['graph'] or 0):.0f}/{(b['llm'] or 0):.0f}")
    for t, row in agg["by_type"].items():
        print(f"  {t:15s} " + "  ".join(f"{e}: rcr={v['rcr']} corr={v['correctness']} (n={v['n']})" for e, v in row.items()))
    er = res["er_resolution"]
    print(f"ER resolution: group_acc={er['group_accuracy']} ({er['groups_correct']}/{er['groups_evaluated']}) link_recall={er['link_recall']}")
    print("spec expectation matches:", {k: v["match"] for k, v in res["spec_comparison"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--engines", nargs="+", choices=config.ENGINES)
    ap.add_argument("--types", nargs="+", choices=config.QUESTION_TYPES)
    a = ap.parse_args()
    res = run_benchmark(a.engines, a.types, a.limit, progress=lambda f, m: print(f"[{f:4.0%}] {m}"))
    print_summary(res)
