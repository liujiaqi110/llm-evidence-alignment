"""S42 / S43 双种子汇总（只读；不重新生成、不改选择结果）。

    python -m rag_finetuning.experiments.p6_v5.scripts.summarize_s42_s43

输入：results/hard_dev_s43/（run_eval_v5 analyze，含 A / B / C42 / C43 / D / S42 / S43，评分器 4.0.3）、
      results/s42/、results/s43/（各自的 comparison.json、per_epoch_metrics.json）
输出 results/s42_s43/：dual_seed_summary.json、dual_seed_table.md、per_item_two_seeds.csv
两个种子的结果都完整报告（含负向结果）；所有结果 provisional。
"""
from __future__ import annotations

import csv
import json

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as sv2
from rag_finetuning.experiments.p6_v5.scripts.analyze_s42 import KEY_METRICS, V5, fmt, metrics, score
from rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 import METRICS

AN = V5 / "results" / "hard_dev_s43"
OUT = V5 / "results" / "s42_s43"
BASES = {"A": "3B Base", "B": "3B v3", "C42": "3B v4 s42", "C43": "3B v4 s43"}
SEEDS = ["S42", "S43"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")
    order = [r["id"] for r in data]
    recs = {r["id"]: r for r in data}
    groups = list(BASES) + ["D"] + SEEDS
    gens = {g: {x["id"]: x["answer"] for x in read_jsonl(AN / f"generations_{g}.jsonl")} for g in groups}
    rows = {g: [{"rec": recs[q], "s": score(recs[q], gens[g][q]), "answer": gens[g][q]} for q in order] for g in groups}
    m = {g: metrics(rows[g]) for g in groups}
    # S42 的 hard_dev_s43 重新汇总应与 results/hard_dev_s42 一致（同一份生成、同一评分器）
    s42_old = {x["id"]: x["answer"] for x in read_jsonl(V5 / "results/hard_dev_s42/generations_S42.jsonl")}
    consistency = sum(s42_old[q] == gens["S42"][q] for q in order)
    paired = {}
    for s in SEEDS:
        for b in BASES:
            paired[f"{b}_vs_{s}"] = {k: sv2.paired_cluster([{"rec": x["rec"], "s": {"v": x["s"][key]}} for x in rows[b]],
                                                           [{"rec": x["rec"], "s": {"v": x["s"][key]}} for x in rows[s]], "v", sl, lower_is_better=low)
                                     for k, (sl, key, low) in METRICS.items() if k in KEY_METRICS and key != "required_fact_coverage"}
    seed_mean = {k: round(sum(m[s][k]["rate"] for s in SEEDS) / 2, 4) for k in KEY_METRICS}
    v4_mean = {k: round((m["C42"][k]["rate"] + m["C43"][k]["rate"]) / 2, 4) for k in KEY_METRICS}
    per = {s: json.loads((V5 / f"results/s{s[1:]}/per_epoch_metrics.json").read_text(encoding="utf-8")) for s in SEEDS}
    cmp = {s: json.loads((V5 / f"results/s{s[1:]}/comparison.json").read_text(encoding="utf-8")) for s in SEEDS}
    summ = {"scorer": "4.0.3", "status": "provisional（人工审核未完成的协议偏离保留）",
            "groups": {g: {k: fmt(m[g][k]) for k in KEY_METRICS} for g in groups},
            "rates": {g: {k: m[g][k]["rate"] for k in KEY_METRICS} for g in groups},
            "seed_mean_S": seed_mean, "seed_mean_v4_C": v4_mean,
            "selected_epoch": {s: per[s]["selection_in_training"]["selected_epoch"] for s in SEEDS},
            "selection_per_epoch": {s: per[s]["selection_in_training"]["per_epoch"] for s in SEEDS},
            "gates_dev_reference_only": {s: cmp[s]["gates_dev_reference_only"] for s in SEEDS},
            "paired": paired, "consistency_S42_generations_hard_dev_s42_vs_s43": f"{consistency}/{len(order)}"}
    (OUT / "dual_seed_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    # 表
    lines = ["| 指标 | A | B | C42 | C43 | D | S42 | S43 | S 均值 | v4 均值 |", "|" + "---|" * 10]
    for k in KEY_METRICS:
        lines.append(f"| {k} | " + " | ".join(fmt(m[g][k]) for g in ["A", "B", "C42", "C43", "D", "S42", "S43"]) +
                     f" | {seed_mean[k]:.3f} | {v4_mean[k]:.3f} |")
    lines += ["", "| 配对（S − 对照） | 指标 | 差值 | 聚类 bootstrap 95% CI |", "|---|---|---|---|"]
    for key, v in paired.items():
        for k in KEY_METRICS[:2] + [KEY_METRICS[3], KEY_METRICS[4], KEY_METRICS[6], KEY_METRICS[8]]:
            if k in v:
                lines.append(f"| {key} | {k} | {v[k]['diff']:+.3f} | [{v[k]['ci95_cluster_bootstrap'][0]:+.3f}, {v[k]['ci95_cluster_bootstrap'][1]:+.3f}] |")
    (OUT / "dual_seed_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 逐题
    with open(OUT / "per_item_two_seeds.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["id", "behavior", "is_multi"] + [f"{g}_{x}" for g in ["A", "B", "C42", "C43", "S42", "S43"] for x in ("multi", "content")] + ["S_agree_multi", "S_agree_content"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(order):
            row = {"id": q, "behavior": recs[q]["behavior"], "is_multi": rows["S42"][i]["s"]["is_multi_v5"]}
            for g in ["A", "B", "C42", "C43", "S42", "S43"]:
                row[f"{g}_multi"] = rows[g][i]["s"]["multi_complete_v5"]
                row[f"{g}_content"] = int(rows[g][i]["s"]["content_ok_v4"])
            row["S_agree_multi"] = row["S42_multi"] == row["S43_multi"]
            row["S_agree_content"] = row["S42_content"] == row["S43_content"]
            w.writerow(row)
    print("\n".join(lines))
    print("selected", summ["selected_epoch"], "S42 consistency", summ["consistency_S42_generations_hard_dev_s42_vs_s43"])
    for s in SEEDS:
        for k, v in summ["gates_dev_reference_only"][s].items():
            print(s, "GATE", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
