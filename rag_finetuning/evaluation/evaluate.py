"""Score saved generations and compare models item-by-item.

    python -m rag_finetuning.evaluation.evaluate --split test \
        --run base=<dir> --run qlora=<dir> [--run ref7b=<dir>] --pair base,qlora --out rag_finetuning/reports/<name>

Outputs (all derived from per-question rule scores; nothing here is human- or LLM-judged unless --judge is given):
  summary.json          aggregate metrics per model (+ Wilson 95% CI), paired comparison, latency / memory
  per_question.csv      question_id, behavior, evidence ids, reference, each model's answer + scores + errors,
                        review_status (provisional until a human reviews the row)
  changes.json          improved / regressed / unchanged items between the paired models with error-type diffs
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from rag_finetuning.common import BUILT, read_jsonl
from rag_finetuning.evaluation.scoring import SCORER_VERSION, aggregate, paired, score_item

METRIC_KEYS = ["pass_rate", "content_ok_rate", "faithfulness_rule", "answer_completeness", "citation_valid_rate", "refusal_accuracy",
               "false_refusal_rate", "multi_evidence_full_coverage", "format_compliance", "item_format_compliance",
               "conflict_handling", "partial_handling", "injection_resistance"]


def load_run(d: Path):
    gens = {g["id"]: g for g in read_jsonl(d / "generations.jsonl")}
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return gens, meta


def latency(gens):
    lat = [g["latency_ms"] for g in gens.values() if g.get("latency_ms") is not None]
    toks = [g["output_tokens"] for g in gens.values() if g.get("output_tokens")]
    mem = [g.get("gpu_peak_alloc_mib") or g.get("gpu_used_total_mib") for g in gens.values()]
    mem = [m for m in mem if m]
    lat_sorted = sorted(lat)
    return {"latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_p95": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 1) if lat else None,
            "output_tokens_mean": round(statistics.mean(toks), 1) if toks else None,
            "tokens_per_s": round(sum(toks) / (sum(lat) / 1000), 1) if lat and toks else None,
            "hit_max_tokens": sum(1 for g in gens.values() if g.get("hit_max_tokens")),
            "gpu_peak_mib": max(mem) if mem else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=None)
    ap.add_argument("--run", action="append", required=True, help="name=dir")
    ap.add_argument("--pair", default=None, help="baseline,candidate for paired comparison")
    ap.add_argument("--review-sheet", default=None, help="CSV with id,review_status to carry human review status")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = read_jsonl(Path(a.data) if a.data else BUILT / f"{a.split}.jsonl")
    review = {}
    if a.review_sheet and Path(a.review_sheet).exists():
        with open(a.review_sheet, encoding="utf-8-sig") as f:
            review = {r["id"]: r.get("review_status") or "provisional" for r in csv.DictReader(f)}
    runs = {}
    for spec in a.run:
        name, d = spec.split("=", 1)
        gens, meta = load_run(Path(d))
        missing = [r["id"] for r in recs if r["id"] not in gens]
        if missing:
            raise SystemExit(f"{name}: missing generations for {missing[:5]}")
        rows = [{"rec": r, "s": score_item(r, gens[r["id"]]["answer"]), "g": gens[r["id"]]} for r in recs]
        runs[name] = {"meta": meta, "rows": rows, "agg": aggregate(rows), "lat": latency(gens)}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"split": a.split, "n_items": len(recs),
               "review_status": dict(Counter(review.get(r["id"], "provisional") for r in recs)),
               "scoring": f"rule-based scorer v{SCORER_VERSION} (evaluation/scoring.py); provisional until human review",
               "models": {n: {"meta": {k: v for k, v in R["meta"].items() if k != "ollama_info"},
                              "metrics": R["agg"], "efficiency": R["lat"]} for n, R in runs.items()}}

    changes = {}
    if a.pair:
        b, c = a.pair.split(",")
        B, C = runs[b]["rows"], runs[c]["rows"]
        comp = {"pass": paired([x["s"]["pass"] for x in B], [x["s"]["pass"] for x in C])}
        for key in ("content_ok", "behavior_ok", "faithful_rule", "format_ok", "false_refusal"):
            comp[key] = paired([bool(x["s"][key]) for x in B], [bool(x["s"][key]) for x in C])
        comp["note"] = ("McNemar exact p and bootstrap CI are reported for transparency; with n<100 and provisional "
                        "labels they are NOT a claim of statistical significance.")
        summary["paired"] = {"baseline": b, "candidate": c, **comp}
        imp, reg, same = [], [], []
        for x, y in zip(B, C):
            item = {"id": x["rec"]["id"], "type": x["rec"]["type"], "behavior": x["rec"]["behavior"],
                    "question": x["rec"]["question"], f"{b}_errors": x["s"]["errors"], f"{c}_errors": y["s"]["errors"],
                    "errors_fixed": sorted(set(x["s"]["errors"]) - set(y["s"]["errors"])),
                    "errors_introduced": sorted(set(y["s"]["errors"]) - set(x["s"]["errors"])),
                    f"{b}_answer": x["g"]["answer"], f"{c}_answer": y["g"]["answer"]}
            (imp if (not x["s"]["pass"] and y["s"]["pass"]) else reg if (x["s"]["pass"] and not y["s"]["pass"])
             else same).append(item)
        changes = {"improved": imp, "regressed": reg, "unchanged": same,
                   "error_counts": {n: dict(Counter(e for r in runs[n]["rows"] for e in r["s"]["errors"])) for n in (b, c)}}
        (out / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["error_counts"] = {n: dict(Counter(e for r in R["rows"] for e in r["s"]["errors"])) for n, R in runs.items()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["question_id", "type", "subtype", "behavior", "generation_method", "question", "evidence_ids", "gold_ids",
            "reference_answer"]
    for n in runs:
        cols += [f"{n}_answer", f"{n}_pass", f"{n}_behavior_ok", f"{n}_pred_behavior", f"{n}_fact_recall",
                 f"{n}_faithful_rule", f"{n}_citation_precision", f"{n}_citation_recall", f"{n}_fabricated",
                 f"{n}_format_ok", f"{n}_errors", f"{n}_latency_ms"]
    cols += ["review_status", "human_verdict", "reviewer_note"]
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(recs):
            row = {"question_id": r["id"], "type": r["type"], "subtype": r["subtype"], "behavior": r["behavior"],
                   "generation_method": r["generation_method"], "question": r["question"],
                   "evidence_ids": " ".join(e["cid"] for e in r["evidence"]), "gold_ids": " ".join(r["gold_cids"]),
                   "reference_answer": r["reference_answer"], "review_status": review.get(r["id"], "provisional"),
                   "human_verdict": "", "reviewer_note": ""}
            for n, R in runs.items():
                s, g = R["rows"][i]["s"], R["rows"][i]["g"]
                row.update({f"{n}_answer": g["answer"], f"{n}_pass": int(s["pass"]), f"{n}_behavior_ok": int(s["behavior_ok"]),
                            f"{n}_pred_behavior": s["pred_behavior"], f"{n}_fact_recall": s["fact_recall"],
                            f"{n}_faithful_rule": int(s["faithful_rule"]), f"{n}_citation_precision": s["citation_precision"],
                            f"{n}_citation_recall": s["citation_recall"], f"{n}_fabricated": " ".join(s["fabricated_citations"]),
                            f"{n}_format_ok": int(s["format_ok"]), f"{n}_errors": ";".join(s["errors"]),
                            f"{n}_latency_ms": g.get("latency_ms")})
            w.writerow(row)

    # console table
    names = list(runs)
    print(f"{'metric':32s}" + "".join(f"{n:>22s}" for n in names))
    for k in METRIC_KEYS:
        cells = []
        for n in names:
            m = runs[n]["agg"][k]
            cells.append(f"{m['rate']:.3f} ({m['k']}/{m['n']})" if m["rate"] is not None else "n/a")
        print(f"{k:32s}" + "".join(f"{c:>22s}" for c in cells))
    for k in ("citation_precision_mean", "citation_recall_mean", "fact_recall_mean"):
        print(f"{k:32s}" + "".join(f"{runs[n]['agg'][k]['mean']!s:>22s}" for n in names))
    for k in ("latency_ms_p50", "tokens_per_s", "gpu_peak_mib", "hit_max_tokens"):
        print(f"{k:32s}" + "".join(f"{runs[n]['lat'][k]!s:>22s}" for n in names))
    if a.pair:
        p = summary["paired"]["pass"]
        print("PAIRED pass:", json.dumps(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
