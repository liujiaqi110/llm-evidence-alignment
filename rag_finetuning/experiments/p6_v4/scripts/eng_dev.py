"""在开发数据上评估并标定两个工程基线（不使用 test_v4）。

    python -m rag_finetuning.experiments.p6_v4.scripts.eng_dev

方案二（冲突预检测）：带标签的片段对来自 v3 test_v3（已看过，只用于开发）与 v4 dev（程序构造，标签由构造给出）。
  正例 = 每题两个金标准片段，标签 = conflict_kind（same_conclusion_despite_values 在证据层面按 conflict_true 计）；
  负例 = 同一题证据中的其余片段对，标签 = none。网格搜索 TAU / TAU_LOW，取宏平均 F1 最大者。
方案一（事实覆盖检查）：对 v3 test_v3 上 D（3B Base）、E（3B v3）的已有输出做后处理，比较多证据内容完整、陈述支持、
  错误拒答的变化，网格搜索 THETA_REL。
结果写入 results/eng_dev.json；选定阈值需手工写回 eng.py 并在 freeze.json 中记录。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v4.scripts import eng
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import score_item_v3

V4 = Path(__file__).resolve().parents[1]
V3 = V4.parent / "p6_v3"
KINDS = ["conflict_true", "version_override", "scope_difference", "condition_difference"]


def labeled_pairs(recs):
    out = []
    for r in recs:
        kind = r.get("conflict_kind")
        if not kind and r["behavior"] == "conflict":
            kind = "conflict_true"
        if not kind and r.get("no_conflict"):
            kind = r.get("subtype")
        if not kind:
            continue
        if kind == "same_conclusion_despite_values":
            kind = "conflict_true"
        if kind == "old_rule_still_applies":
            kind = "version_override"
        gold = set(r["gold_cids"])
        for a, b in combinations([e["cid"] for e in r["evidence"]], 2):
            lab = kind if {a, b} == gold else "none"
            out.append((r, tuple(sorted((a, b))), lab))
    return out


def eval_detector(pairs, tau, tau_low):
    cache, conf = {}, Counter()
    for r, pair, lab in pairs:
        if r["id"] not in cache:
            cache[r["id"]] = {tuple(d["cids"]): d["kind"] for d in eng.detect_conflicts(r["evidence"], tau, tau_low)}
        pred = cache[r["id"]].get(pair, "none")
        if pred == "insufficient":
            pred = "none"
        conf[(lab, pred)] += 1
    f1s, per = [], {}
    for k in KINDS:
        tp = conf[(k, k)]
        fp = sum(v for (l, p), v in conf.items() if p == k and l != k)
        fn = sum(v for (l, p), v in conf.items() if l == k and p != k)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[k] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)}
        f1s.append(f1)
    none_fp = sum(v for (l, p), v in conf.items() if l == "none" and p != "none")
    return {"macro_f1": round(sum(f1s) / len(f1s), 3), "per_class": per, "none_false_alarms": none_fp,
            "n_none": sum(v for (l, _), v in conf.items() if l == "none"), "confusion": {f"{l}->{p}": v for (l, p), v in conf.items()}}


def main() -> int:
    v3test = read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl")
    v4dev = [r for r in read_jsonl(V4 / "data" / "v4" / "dev.jsonl") if r.get("generation_method") == "programmatic_structured"]
    pairs = labeled_pairs(v3test) + labeled_pairs(v4dev)
    grid = []
    for tau in (0.35, 0.45, 0.55, 0.65, 0.75):
        for tl in (0.2, 0.3):
            if tl < tau:
                res = eval_detector(pairs, tau, tl)
                grid.append({"tau": tau, "tau_low": tl, "macro_f1": res["macro_f1"], "none_false_alarms": res["none_false_alarms"]})
    best = max(grid, key=lambda g: (g["macro_f1"], -g["none_false_alarms"], -g["tau"]))
    det = eval_detector(pairs, best["tau"], best["tau_low"])
    det["by_source"] = {"v3_test": eval_detector(labeled_pairs(v3test), best["tau"], best["tau_low"]),
                        "v4_dev": eval_detector(labeled_pairs(v4dev), best["tau"], best["tau_low"])}
    # 方案一：对 v3 test 上 D / E 的已有输出做后处理
    cov = {}
    for g in "DE":
        gens = {x["id"]: x["answer"] for x in read_jsonl(Path(f"workspace/runs/p6_v3/eval/test_v3/{g}/generations.jsonl"))}
        cov[g] = {}
        for th in (0.2, 0.3, 0.4, 0.5):
            agg = defaultdict(int)
            for r in v3test:
                a0 = gens[r["id"]]
                a1 = eng.coverage_check(r["question"], r["evidence"], a0, th)["answer"]
                for tag, a in (("before", a0), ("after", a1)):
                    s = score_item_v3(r, a)
                    agg[f"{tag}_content_ok"] += int(s["content_ok"])
                    agg[f"{tag}_multi_content"] += int(bool(s["multi_content_complete"]))
                    agg[f"{tag}_multi_cov"] += int(bool(s["multi_chunk_coverage"]))
                    agg[f"{tag}_support_full"] += int(bool(s["claim_support_full"]))
                    agg[f"{tag}_refusal_ok"] += int(bool(s["refusal_ok"]))
                    agg[f"{tag}_whole_refusal"] += int(bool(s["whole_refusal_on_answerable"]))
                agg["n_modified"] += int(a1 != a0)
            agg["n_multi"] = sum(1 for r in v3test if score_item_v3(r, r["reference_answer"])["is_multi"])
            cov[g][str(th)] = dict(agg)
    out = {"detector_grid": grid, "detector_best": best, "detector_eval_at_best": det, "coverage_dev": cov}
    (V4 / "results").mkdir(exist_ok=True)
    (V4 / "results" / "eng_dev.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("best", best)
    print(json.dumps({k: det[k] for k in ("macro_f1", "per_class", "none_false_alarms", "n_none")}, ensure_ascii=False))
    print({s: v["macro_f1"] for s, v in det["by_source"].items()})
    for g in cov:
        for th, a in cov[g].items():
            print(g, th, {k: a[k] for k in ("n_modified", "before_content_ok", "after_content_ok", "before_multi_content", "after_multi_content",
                                            "before_support_full", "after_support_full", "before_refusal_ok", "after_refusal_ok")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
