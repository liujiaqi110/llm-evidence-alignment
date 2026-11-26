"""P6-v4 回归：v4 适配器（C42 / C43）在 P6-v3 冻结测试集 test_v3（81 题）上，与 v3 已有输出（D 3B Base、E 3B v3、F 7B）对比。
评分器 v3（与 test_v4 相同）；同时报告评分器 v2 的 content_ok 以便与 P6-v3 报告对照。输入逐题 input_sha256 必须一致。

    python -m rag_finetuning.experiments.p6_v4.scripts.regression_v4
"""
from __future__ import annotations

import json
from pathlib import Path

from rag_finetuning.common import REPO, read_jsonl
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as sv2
from rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 import METRICS
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import score_item_v3

V4 = Path(__file__).resolve().parents[1]
REF = REPO / "rag_finetuning/experiments/p6_v3/data/test_v3/built/test_v3.jsonl"
MAN = V4 / "data/frozen/reg_test_v3.manifest.json"
SRC = {"D(3B Base)": "workspace/runs/p6_v3/eval/test_v3/D",
       "E(3B v3)": "workspace/runs/p6_v3/eval/test_v3/E",
       "F(7B)": "workspace/runs/p6_v3/eval/test_v3/F",
       "C42(3B v4 s42)": "workspace/runs/p6_v4/eval/reg_test_v3/C42",
       "C43(3B v4 s43)": "workspace/runs/p6_v4/eval/reg_test_v3/C43"}
PAIRS = [("E(3B v3)", "C42(3B v4 s42)"), ("E(3B v3)", "C43(3B v4 s43)"), ("D(3B Base)", "C42(3B v4 s42)"), ("D(3B Base)", "C43(3B v4 s43)")]


def main() -> int:
    man = json.loads(MAN.read_text(encoding="utf-8"))
    recs = {r["id"]: r for r in read_jsonl(REF)}
    order = list(man["items"])
    data = {}
    for g, d in SRC.items():
        gens = {x["id"]: x for x in read_jsonl(Path(d) / "generations.jsonl")}
        mism = [q for q in order if gens[q]["input_sha256"] != man["items"][q]]
        assert not mism, (g, mism[:3])
        data[g] = [{"rec": recs[q], "s": score_item_v3(recs[q], gens[q]["answer"])} for q in order]
    out = {"input": "reg_test_v3", "n": len(order), "frozen_sha256": man["frozen_sha256"], "scorer": "v3",
           "label": "自动规则评分，provisional（未经人工审核）", "groups": {}, "pairs": {}}
    for g, rows in data.items():
        m = {k: sv2.rate(x["s"][key] for x in rows if sel(x["rec"])) for k, (sel, key, low) in METRICS.items()}
        m["v2_content_ok"] = sv2.rate(x["s"]["v2"]["content_ok"] for x in rows)
        out["groups"][g] = m
    for x, y in PAIRS:
        out["pairs"][f"{x}_vs_{y}"] = {k: sv2.paired_cluster([{"rec": r["rec"], "s": {"v": r["s"][key]}} for r in data[x]],
                                                             [{"rec": r["rec"], "s": {"v": r["s"][key]}} for r in data[y]], "v", sel,
                                                             lower_is_better=low) for k, (sel, key, low) in METRICS.items()}
    p = V4 / "results/regression_test_v3/summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ["content_ok", "v2_content_ok", "fact_complete(应回答类)", "multi_content_complete", "whole_refusal(应回答类)", "false_refusal(answer 题)",
            "refusal_ok", "partial_ok", "claim_support_full(全分母)", "conflict_true_ok", "format_ok", "citation_valid"]
    print(f"{'指标':28s}" + "".join(f"{g[:12]:>15s}" for g in data))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{out['groups'][g][k]['k']}/{out['groups'][g][k]['n']}".rjust(15) for g in data))
    for p_, M in out["pairs"].items():
        print(p_, {k: (M[k].get("diff"), M[k].get("ci95_cluster_bootstrap")) for k in ("content_ok", "multi_content_complete",
                                                                                         "whole_refusal(应回答类)", "refusal_ok", "claim_support_full(全分母)")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
