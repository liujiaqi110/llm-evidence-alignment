"""P6-v2 统计分析：多组逐题对照、逐指标配对差值（bootstrap 95% CI + exact McNemar）、错误案例分类、成本。

    python -m rag_finetuning.experiments.p6_v2.analyze --set diagnostic_47 \
        --ref rag_finetuning/data/built/test.jsonl --runs workspace/runs/p6_v2/diagnostic_47 \
        --out rag_finetuning/experiments/p6_v2/results/diagnostic_47

评分全部是自动规则评分（evaluation/scoring.py v1.1），未经人工审核 → provisional。
参考答案只在这里（评分阶段）读取，推理阶段读的是不含参考答案的冻结输入。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

import yaml

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.scoring import SCORER_VERSION, aggregate, mcnemar_exact, score_item

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
ANSWERABLE = ("answer", "partial", "conflict")

# 指标名 → (适用题目, 取值函数)。取值为 bool 或 float；None 表示该题不适用
METRICS = {
    "content_ok": (lambda r: True, lambda s: s["content_ok"]),
    "answer_complete": (lambda r: r["behavior"] in ANSWERABLE, lambda s: s["complete"]),
    "faithful_rule": (lambda r: True, lambda s: s["faithful_rule"]),
    "citation_precision": (lambda r: r["behavior"] in ANSWERABLE, lambda s: s["citation_precision"]),
    "citation_recall": (lambda r: r["behavior"] in ANSWERABLE, lambda s: s["citation_recall"]),
    "no_fabricated_citation": (lambda r: True, lambda s: not s["fabricated_citations"]),
    "format_ok": (lambda r: True, lambda s: s["format_ok"]),
    "refusal_correct": (lambda r: r["behavior"] == "refuse", lambda s: s["behavior_ok"]),
    "false_refusal(↓)": (lambda r: r["behavior"] == "answer", lambda s: s["false_refusal"]),
    "multi_evidence_full": (lambda r: len(r["gold_cids"]) >= 2 and r["behavior"] != "refuse",
                            lambda s: s["citation_recall"] == 1.0 and s["complete"] is not False),
    "conflict_correct": (lambda r: r["behavior"] == "conflict", lambda s: s["behavior_ok"]),
    "partial_correct": (lambda r: r["behavior"] == "partial", lambda s: s["behavior_ok"]),
    "injection_resisted": (lambda r: r.get("subtype") == "prompt_injection",
                           lambda s: s["behavior_ok"] and not s["fabricated_citations"] and not s["forbidden_hits"]),
}


def boot_ci(diffs, iters=10000, seed=0):
    n = len(diffs)
    if n == 0:
        return None
    rng = random.Random(seed)
    m = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return [round(m[int(0.025 * iters)], 4), round(m[int(0.975 * iters) - 1], 4)]


def metric_values(rows, name):
    sel, f = METRICS[name]
    return {x["rec"]["id"]: f(x["s"]) for x in rows if sel(x["rec"]) and f(x["s"]) is not None}


def summarize(rows, name):
    v = metric_values(rows, name)
    vals = list(v.values())
    if not vals:
        return {"n": 0}
    if all(isinstance(x, bool) for x in vals):
        return {"k": sum(vals), "n": len(vals), "rate": round(sum(vals) / len(vals), 4)}
    return {"n": len(vals), "mean": round(statistics.mean(vals), 4)}


def compare(ra, rb, name):
    va, vb = metric_values(ra, name), metric_values(rb, name)
    ids = sorted(set(va) & set(vb))
    if not ids:
        return {"n": 0}
    a, b = [float(va[i]) for i in ids], [float(vb[i]) for i in ids]
    d = [y - x for x, y in zip(a, b)]
    out = {"n": len(ids), "a": round(sum(a) / len(ids), 4), "b": round(sum(b) / len(ids), 4),
           "diff": round(sum(d) / len(ids), 4), "ci95": boot_ci(d)}
    if all(x in (0.0, 1.0) for x in a + b):
        up = [i for i in ids if not va[i] and vb[i]]
        down = [i for i in ids if va[i] and not vb[i]]
        out.update({"a_k": int(sum(a)), "b_k": int(sum(b)), "b_better": up if not name.endswith("(↓)") else down,
                    "b_worse": down if not name.endswith("(↓)") else up, "mcnemar_p": mcnemar_exact(len(up), len(down))})
    return out


def cost(gens):
    lat = sorted(g["latency_ms"] for g in gens)
    tin = [g["input_tokens_hf"] for g in gens]
    tout = [g["output_tokens"] for g in gens if g.get("output_tokens") is not None]
    oc = [g.get("ollama_prompt_eval_count") for g in gens if g.get("ollama_prompt_eval_count") is not None]
    return {"latency_ms_p50": round(statistics.median(lat), 1), "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "latency_ms_total": round(sum(lat), 1),
            "input_tokens_hf_mean": round(statistics.mean(tin), 1), "input_tokens_hf_total": sum(tin),
            "output_tokens_mean": round(statistics.mean(tout), 1) if tout else None,
            "output_tokens_total": sum(tout) if tout else None,
            "ollama_prompt_eval_count_mean": round(statistics.mean(oc), 1) if oc else None,
            "hit_max_tokens": sum(1 for g in gens if g.get("hit_max_tokens")),
            "token_note": "input_tokens_hf 为统一口径；ollama_prompt_eval_count 受前缀缓存影响，仅供参考"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ref", required=True, help="含参考答案的构建文件（只用于评分）")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--groups", default="A,B,C,D,E")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    manifest = json.loads((HERE / "frozen_inputs" / f"{a.set}.manifest.json").read_text(encoding="utf-8"))
    recs = read_jsonl(Path(a.ref))
    by_id = {r["id"]: r for r in recs}
    groups = [g for g in a.groups.split(",") if (Path(a.runs) / g / "generations.jsonl").exists()]
    data = {}
    for g in groups:
        gens = {x["id"]: x for x in read_jsonl(Path(a.runs) / g / "generations.jsonl")}
        meta = json.loads((Path(a.runs) / g / "meta.json").read_text(encoding="utf-8"))
        assert meta["frozen_file_sha256"] == manifest["frozen_file_sha256"], f"{g}: 输入快照不一致"
        for qid, x in gens.items():
            assert x["input_sha256"] == manifest["items"][qid], f"{g}/{qid}: 题目输入哈希不一致"
        rows = [{"rec": by_id[q], "s": score_item(by_id[q], gens[q]["answer"]), "g": gens[q]} for q in manifest["items"]]
        data[g] = {"meta": meta, "rows": rows, "gens": list(gens.values())}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"set": a.set, "n_items": len(manifest["items"]), "frozen_file_sha256": manifest["frozen_file_sha256"],
               "scoring": f"自动规则评分 scorer v{SCORER_VERSION}；未经人工审核（provisional）", "groups": {}}
    for g, D in data.items():
        summary["groups"][g] = {"desc": CFG["groups"][g]["desc"], "model": D["meta"]["model"],
                                "prompt_variant": D["meta"]["prompt_variant"],
                                "metrics": {m: summarize(D["rows"], m) for m in METRICS},
                                "aggregate_v1_metrics": aggregate(D["rows"]), "cost": cost(D["gens"]),
                                "generation_settings": D["meta"]["generation_settings"]}
    pairs = {}
    for c in CFG["comparisons"] + [{"a": "A", "b": "C", "question": "额外：原始 1.5B vs 原始 7B"},
                                   {"a": "B", "b": "D", "question": "额外：QLoRA 1.5B vs 7B Few-shot"}]:
        if c["a"] in data and c["b"] in data:
            pairs[f"{c['a']}_vs_{c['b']}"] = {"question": c["question"],
                                              "metrics": {m: compare(data[c["a"]]["rows"], data[c["b"]]["rows"], m)
                                                          for m in METRICS}}
    summary["pairs"] = pairs
    summary["note"] = "n 较小且未经人工审核：CI 与 McNemar p 仅作透明展示，不构成统计显著性结论"
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐题对照表
    cols = ["question_id", "type", "subtype", "behavior", "question", "evidence_ids", "gold_ids", "reference_answer"]
    for g in data:
        cols += [f"{g}_answer", f"{g}_status", f"{g}_citations", f"{g}_content_ok", f"{g}_complete", f"{g}_faithful_rule",
                 f"{g}_format_ok", f"{g}_citation_precision", f"{g}_errors", f"{g}_input_tokens_hf", f"{g}_output_tokens",
                 f"{g}_latency_ms"]
    cols += ["review_status", "human_verdict", "reviewer_note"]
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(manifest["items"]):
            r = by_id[q]
            row = {"question_id": q, "type": r["type"], "subtype": r["subtype"], "behavior": r["behavior"],
                   "question": r["question"], "evidence_ids": " ".join(e["cid"] for e in r["evidence"]),
                   "gold_ids": " ".join(r["gold_cids"]), "reference_answer": r["reference_answer"],
                   "review_status": "provisional", "human_verdict": "", "reviewer_note": ""}
            for g, D in data.items():
                x = D["rows"][i]
                row.update({f"{g}_answer": x["g"]["answer"], f"{g}_status": x["g"]["status"],
                            f"{g}_citations": " ".join(x["g"]["citations"]), f"{g}_content_ok": int(x["s"]["content_ok"]),
                            f"{g}_complete": x["s"]["complete"], f"{g}_faithful_rule": int(x["s"]["faithful_rule"]),
                            f"{g}_format_ok": int(x["s"]["format_ok"]), f"{g}_citation_precision": x["s"]["citation_precision"],
                            f"{g}_errors": ";".join(x["s"]["errors"]), f"{g}_input_tokens_hf": x["g"]["input_tokens_hf"],
                            f"{g}_output_tokens": x["g"].get("output_tokens"), f"{g}_latency_ms": x["g"]["latency_ms"]})
            w.writerow(row)

    # 错误案例
    cases = {}
    for g, D in data.items():
        rows = D["rows"]
        brief = lambda x: {"id": x["rec"]["id"], "behavior": x["rec"]["behavior"], "question": x["rec"]["question"],  # noqa: E731
                           "answer": x["g"]["answer"], "errors": x["s"]["errors"]}
        cases[g] = {
            "false_refusal": [brief(x) for x in rows if x["s"]["false_refusal"]],
            "format_ok_but_irrelevant_citation": [dict(brief(x), cited=x["s"]["valid_citations"], gold=x["rec"]["gold_cids"],
                                                       citation_precision=x["s"]["citation_precision"])
                                                  for x in rows if x["rec"]["behavior"] in ANSWERABLE and x["s"]["format_ok"]
                                                  and x["s"]["valid_citations"] and (x["s"]["citation_precision"] or 0) < 1.0],
            "conflict_failed": [brief(x) for x in rows if x["rec"]["behavior"] == "conflict" and not x["s"]["behavior_ok"]],
            "injection_failed": [brief(x) for x in rows if x["rec"].get("subtype") == "prompt_injection"
                                 and not (x["s"]["behavior_ok"] and not x["s"]["forbidden_hits"])],
            "missed_refusal": [brief(x) for x in rows if x["rec"]["behavior"] == "refuse" and not x["s"]["behavior_ok"]],
        }
    for key, P in pairs.items():
        ga, gb = key.split("_vs_")
        ra = {x["rec"]["id"]: x for x in data[ga]["rows"]}
        rb = {x["rec"]["id"]: x for x in data[gb]["rows"]}
        cases[f"regressed_{key}"] = [{"id": q, "question": ra[q]["rec"]["question"], "behavior": ra[q]["rec"]["behavior"],
                                      f"{ga}_answer": ra[q]["g"]["answer"], f"{gb}_answer": rb[q]["g"]["answer"],
                                      f"{gb}_errors": rb[q]["s"]["errors"]}
                                     for q in P["metrics"]["content_ok"].get("b_worse", [])]
        cases[f"improved_{key}"] = [{"id": q, "question": ra[q]["rec"]["question"], "behavior": ra[q]["rec"]["behavior"],
                                     f"{ga}_answer": ra[q]["g"]["answer"], f"{gb}_answer": rb[q]["g"]["answer"],
                                     f"{ga}_errors": ra[q]["s"]["errors"]}
                                    for q in P["metrics"]["content_ok"].get("b_better", [])]
    (out / "error_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    # 逐组原始输出（非敏感合成数据，体积小）
    for g, D in data.items():
        (out / f"generations_{g}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in D["gens"]),
                                                    encoding="utf-8")

    # 控制台
    gs = list(data)
    print(f"{'指标':26s}" + "".join(f"{g:>16s}" for g in gs))
    for m in METRICS:
        cells = []
        for g in gs:
            s = summary["groups"][g]["metrics"][m]
            cells.append(f"{s['k']}/{s['n']}" if "k" in s else (f"{s['mean']:.3f}(n={s['n']})" if s.get("n") else "-"))
        print(f"{m:26s}" + "".join(f"{c:>16s}" for c in cells))
    for k in ("input_tokens_hf_mean", "output_tokens_mean", "latency_ms_p50", "latency_ms_p95"):
        print(f"{k:26s}" + "".join(f"{summary['groups'][g]['cost'][k]!s:>16s}" for g in gs))
    for key, P in pairs.items():
        m = P["metrics"]
        print(key, {n: (m[n].get("diff"), m[n].get("ci95"), m[n].get("mcnemar_p")) for n in
                    ("content_ok", "answer_complete", "false_refusal(↓)", "refusal_correct", "citation_precision")})
    print("hash", hashlib.sha256((out / "summary.json").read_bytes()).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
