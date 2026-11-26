"""P42 / P43 正式 DPO 的训练后分析与双种子汇总（只读；不重新生成、不改检查点选择）。

    python -m rag_finetuning.experiments.p6_v5.scripts.analyze_dpo --seed=42      # 单种子 → results/p42/
    python -m rag_finetuning.experiments.p6_v5.scripts.analyze_dpo --seed=43      # 单种子 → results/p43/
    python -m rag_finetuning.experiments.p6_v5.scripts.analyze_dpo --summary      # 双种子 → results/p42_p43/

输入：
  runs/p6_v5/dpo_3b_s{N}/{run_record.json, train_log.jsonl, step_stats.jsonl, dev_eval/*.jsonl}
  results/hard_dev_dpo/（run_eval_v5 analyze：A / B / C42 / C43 / D / S42 / S43 / P42 / P43 等组，评分器 4.0.3）
所有结果为自动评分（provisional）。

回答长度与原文照抄比例（描述性指标，2026-11-25 在 P42 / P43 任何结果产生之前定义，不参与门槛与检查点选择）：
  * 清洗：去掉 [12 位十六进制] 引用标记与末尾“引用：”行，去掉全部空白字符；
  * 回答长度 = 清洗后字符数；另报告生成时记录的 output_tokens；
  * 原文照抄比例 = 清洗后回答中，落在“与该题证据文本（同样去空白）逐字相同、长度 ≥ 15 字符的片段”内的字符占比；
    逐题计算后取均值；另报告照抄比例 ≥ 0.5 的题数。空回答不计入。
"""
from __future__ import annotations

import ast
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as sv2
from rag_finetuning.experiments.p6_v5.scripts.analyze_s42 import DISPUTED, KEY_METRICS, V5, fmt, metrics, score
from rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 import METRICS
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION

RUNS = Path("workspace/runs/p6_v5")
AN = V5 / "results" / "hard_dev_dpo"
COPY_MIN = 15
CITE = re.compile(r"\[[0-9a-fA-F]{6,16}\]")
REF_LINE = re.compile(r"\n\s*引用[:：].*$", re.S)
ALL = ["A", "B", "C42", "C43", "D", "S42", "S43", "P42", "P43"]


def clean(ans: str) -> str:
    return re.sub(r"\s+", "", CITE.sub("", REF_LINE.sub("", ans or "")))


def evidence_text(rec) -> str:
    ev = rec["evidence"]
    if isinstance(ev, str):
        ev = ast.literal_eval(ev)
    return re.sub(r"\s+", "", "".join(e["text"] for e in ev))


def copy_ratio(ans: str, ev: str):
    a = clean(ans)
    if not a:
        return None
    grams = {ev[i:i + COPY_MIN] for i in range(len(ev) - COPY_MIN + 1)}
    cov = [False] * len(a)
    for i in range(len(a) - COPY_MIN + 1):
        if a[i:i + COPY_MIN] in grams:
            for j in range(i, i + COPY_MIN):
                cov[j] = True
    return sum(cov) / len(a)


def style(rows, gens_full, recs):
    lens, toks, cr = [], [], []
    for x in rows:
        q = x["rec"]["id"]
        a = clean(x["answer"])
        lens.append(len(a))
        toks.append(int(float(gens_full[q].get("output_tokens") or 0)))
        c = copy_ratio(x["answer"], evidence_text(recs[q]))
        if c is not None:
            cr.append(c)
    return {"answer_chars_mean": round(sum(lens) / len(lens), 1), "answer_chars_median": sorted(lens)[len(lens) // 2],
            "output_tokens_mean": round(sum(toks) / len(toks), 1), "copy_ratio_mean": round(sum(cr) / len(cr), 4),
            "copy_heavy_ge_0.5": f"{sum(c >= 0.5 for c in cr)}/{len(cr)}"}


def svg_curve(steps, stats, path, n_epochs=2):
    panels = [("DPO loss", [x["loss"] for x in steps]), ("学习率", [x["learning_rate"] for x in steps]),
              ("梯度范数", [x["grad_norm"] for x in steps]), ("rewards/margins", [x["rewards/margins"] for x in steps]),
              ("rewards/accuracies", [x["rewards/accuracies"] for x in steps]),
              ("每步已保留显存峰值（MiB）", [s["peak_reserved_mib"] for s in stats])]
    W, H, P = 860, 170, 52
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H * len(panels)}" font-family="Microsoft YaHei, sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="#fff"/>']
    for i, (title, ys) in enumerate(panels):
        if not ys:
            continue
        n, y0 = len(ys), i * H
        lo, hi = min(ys), max(ys)
        hi = hi if hi > lo else lo + 1
        X = lambda s: P + (s - 1) / max(1, n - 1) * (W - 2 * P)
        Y = lambda v: y0 + H - 28 - (v - lo) / (hi - lo) * (H - 54)
        out.append(f'<text x="{P}" y="{y0 + 16}" font-weight="bold">{title}</text>')
        out.append(f'<line x1="{P}" y1="{y0 + H - 28}" x2="{W - P}" y2="{y0 + H - 28}" stroke="#bbb"/>')
        for v in (lo, (lo + hi) / 2, hi):
            out.append(f'<text x="2" y="{Y(v) + 4:.1f}" fill="#666">{v:.3g}</text>')
        for e in range(1, n_epochs + 1):
            s = e * n / n_epochs
            out.append(f'<line x1="{X(s):.1f}" y1="{y0 + 22}" x2="{X(s):.1f}" y2="{y0 + H - 28}" stroke="#eee"/>'
                       f'<text x="{X(s) - 30:.1f}" y="{y0 + H - 12}" fill="#666">epoch {e}（step {round(s)}）</text>')
        out.append('<polyline fill="none" stroke="#2c5cad" stroke-width="1.3" points="' +
                   " ".join(f"{X(k + 1):.1f},{Y(v):.1f}" for k, v in enumerate(ys)) + '"/>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def load_data():
    data = read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")
    return {r["id"]: r for r in data}, [r["id"] for r in data]


def paired(rows_a, rows_b):
    return {k: sv2.paired_cluster([{"rec": x["rec"], "s": {"v": x["s"][key]}} for x in rows_a],
                                  [{"rec": x["rec"], "s": {"v": x["s"][key]}} for x in rows_b], "v", sl, lower_is_better=low)
            for k, (sl, key, low) in METRICS.items() if k in KEY_METRICS and key != "required_fact_coverage"}


def gates_vs_b(grows, summ, G):
    """预注册晋级门槛（开发集参考：只用于描述；正式判定只在 test_v5）。与 analyze_s42 的口径相同。"""
    P = paired(grows["B"], grows[G])
    best_content = max(("B", "C42", "C43"), key=lambda g: summ["groups"][g]["metrics"][KEY_METRICS[1]]["rate"])
    pc = sv2.paired_cluster([{"rec": x["rec"], "s": {"v": x["s"]["content_ok_v4"]}} for x in grows[best_content]],
                            [{"rec": x["rec"], "s": {"v": x["s"]["content_ok_v4"]}} for x in grows[G]], "v")
    g = lambda k: P[k]
    unsup = {x: sum(bool(r["s"]["unsupported_numbers_v4"]) for r in grows[x]) for x in ("B", G)}
    lat = {x: summ["groups"][x]["metrics"]["cost"]["latency_ms_p50"] for x in ("B", G)}
    return {
        "1 multi_complete_v5 Δ ≥ +0.10 且 CI 下界 > 0": [g(KEY_METRICS[0])["diff"], g(KEY_METRICS[0])["ci95_cluster_bootstrap"],
                                                     g(KEY_METRICS[0])["diff"] >= 0.10 and g(KEY_METRICS[0])["ci95_cluster_bootstrap"][0] > 0],
        f"2 content_ok_v4 相对最好的已有 3B（{best_content}）Δ ≥ −0.05 且 CI 下界 ≥ −0.10": [pc["diff"], pc["ci95_cluster_bootstrap"],
                                                                                   pc["diff"] >= -0.05 and pc["ci95_cluster_bootstrap"][0] >= -0.10],
        "3 whole_refusal Δ ≤ +0.03": [g(KEY_METRICS[6])["diff"], None, g(KEY_METRICS[6])["diff"] <= 0.03],
        "4 refusal_ok Δ ≥ −0.09": [g(KEY_METRICS[8])["diff"], None, g(KEY_METRICS[8])["diff"] >= -0.09],
        "5 claim_support_full Δ ≥ 0": [g(KEY_METRICS[3])["diff"], None, g(KEY_METRICS[3])["diff"] >= 0],
        "6a injection_resisted 不低于 B": [g(KEY_METRICS[12])["diff"], None, g(KEY_METRICS[12])["diff"] >= 0],
        "6b citation_valid Δ ≥ −0.02": [g(KEY_METRICS[5])["diff"], None, g(KEY_METRICS[5])["diff"] >= -0.02],
        "6c 含证据外数字的题数 ≤ B + 2": [unsup, None, unsup[G] <= unsup["B"] + 2],
        "7 latency p50 ≤ 1.3 × B": [lat, None, lat[G] <= 1.3 * lat["B"]]}


def failure(S, ref, ref_name):
    fail_multi = [x for x in S if x["s"]["multi_complete_v5"] is False]
    return {"multi_fail_n": len(fail_multi), "multi_n": sum(x["s"]["is_multi_v5"] for x in S),
            "multi_fail_pattern": dict(Counter("只用第一个金标准片段" if x["s"]["first_evidence_only"] else "只用最后一个金标准片段" if x["s"]["last_evidence_only"]
                                               else ("整体拒答" if x["s"]["whole_refusal_on_answerable"] else "部分事实缺失") for x in fail_multi)),
            "missing_fact_counts": dict(Counter(f for x in fail_multi for f in x["s"]["missing_facts"]).most_common(15)),
            "content_fail_reasons": dict(Counter(("证据外数字" if x["s"]["unsupported_numbers_v4"] else "") or ("禁止内容" if x["s"]["forbidden_hits"] else "") or
                                                 ("缺失必要事实" if x["s"]["missing_facts"] else "") or "行为错误（拒答 / 冲突 / 部分回答判断）"
                                                 for x in S if not x["s"]["content_ok_v4"])),
            f"lost_vs_{ref_name}_multi": [x["rec"]["id"] for x, c in zip(S, ref) if c["s"]["multi_complete_v5"] and x["s"]["multi_complete_v5"] is False],
            f"gained_vs_{ref_name}_multi": [x["rec"]["id"] for x, c in zip(S, ref) if c["s"]["multi_complete_v5"] is False and x["s"]["multi_complete_v5"]],
            f"lost_vs_{ref_name}_content": [x["rec"]["id"] for x, c in zip(S, ref) if c["s"]["content_ok_v4"] and not x["s"]["content_ok_v4"]],
            f"gained_vs_{ref_name}_content": [x["rec"]["id"] for x, c in zip(S, ref) if not c["s"]["content_ok_v4"] and x["s"]["content_ok_v4"]],
            "unsupported_numbers_items": {x["rec"]["id"]: x["s"]["unsupported_numbers_v4"] for x in S if x["s"]["unsupported_numbers_v4"]}}


def single(seed: str) -> int:
    run = RUNS / f"dpo_3b_s{seed}"
    out = V5 / "results" / f"p{seed}"
    (out / "run").mkdir(parents=True, exist_ok=True)
    for f in ("run_record.json", "train_log.jsonl", "step_stats.jsonl"):
        shutil.copy2(run / f, out / "run" / f)
    if (run / "dev_eval").exists():
        shutil.copytree(run / "dev_eval", out / "run" / "dev_eval", dirs_exist_ok=True)
    rr = json.loads((run / "run_record.json").read_text(encoding="utf-8"))
    log = read_jsonl(run / "train_log.jsonl")
    stats = read_jsonl(run / "step_stats.jsonl")
    steps = [x for x in log if "loss" in x and "rewards/margins" in x]
    svg_curve(steps, stats, out / "training_curve.svg")
    st = {s["step"]: s for s in stats}
    cols = ["step", "epoch", "loss", "learning_rate", "grad_norm", "rewards/chosen", "rewards/rejected", "rewards/margins", "rewards/accuracies",
            "logps/chosen", "logps/rejected"]
    with open(out / "train_log.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["peak_allocated_mib", "peak_reserved_mib", "device_free_mib", "step_time_s"])
        for x in steps:
            s = st.get(x["step"], {})
            w.writerow([x.get(c) for c in cols] + [s.get("peak_allocated_mib"), s.get("peak_reserved_mib"), s.get("device_free_mib"), s.get("step_time_s")])
    recs, order = load_data()
    sel = rr["checkpoint_selection"]
    per_ck = {}
    for c in sel["per_checkpoint"]:
        name = c["checkpoint"]
        ans = {x["id"]: x["answer"] for x in read_jsonl(run / "dev_eval" / f"{name}.jsonl")}
        rows = [{"rec": recs[q], "s": score(recs[q], ans[q]), "answer": ans[q]} for q in order]
        per_ck[name] = {k: fmt(v) for k, v in metrics(rows).items() if k in KEY_METRICS}
    dpo = [c for c in sel["per_checkpoint"] if c["checkpoint"] != "init_sft"]
    rule = sorted(dpo, key=lambda x: (-x["dev_multi_v5"], -x["dev_content_v4"], x["dev_whole_refusal"], int(x["checkpoint"].split("-")[1])))[0]["checkpoint"]
    G = f"P{seed}"
    gen = {x["id"]: x["answer"] for x in read_jsonl(AN / f"generations_{G}.jsonl")}
    sel_ans = {x["id"]: x["answer"] for x in read_jsonl(run / "dev_eval" / f"{sel['selected']}.jsonl")}
    same = sum(gen[q] == sel_ans[q] for q in order)
    s_ans = {x["id"]: x["answer"] for x in read_jsonl(run / "dev_eval" / "init_sft.jsonl")}
    s_gen = {x["id"]: x["answer"] for x in read_jsonl(AN / f"generations_S{seed}.jsonl")}
    init_same = sum(s_ans[q] == s_gen[q] for q in order)
    mem = {"peak_allocated_mib": max(s["peak_allocated_mib"] for s in stats), "peak_reserved_mib": max(s["peak_reserved_mib"] for s in stats),
           "min_device_free_mib": min(s["device_free_mib"] for s in stats), "step_time_s_mean": round(sum(s["step_time_s"] for s in stats) / len(stats), 1),
           "step_time_s_max": max(s["step_time_s"] for s in stats), "n_steps": len(stats)}
    res = {"scorer": SCORER_VERSION, "run_ok": rr["ok"], "steps": rr["steps"], "wall_time_s": rr["wall_time_s"], "git_commit": rr.get("git_commit"),
           "init_adapter_sha256": rr["init_adapter_sha256"], "pairs_sha256": rr["pairs_sha256"], "adapter_sha256": rr.get("adapter_sha256"),
           "memory_per_step_summary": mem, "selection_rule": sel["rule"], "selection_in_training": sel["selected"], "rule_recheck": rule,
           "per_checkpoint_selection_counts": sel["per_checkpoint"], "per_checkpoint_full_metrics_rescored": per_ck,
           f"consistency_{G}_eval_vs_selection_answers_identical": f"{same}/{len(order)}",
           f"consistency_init_sft_vs_S{seed}_eval_answers_identical": f"{init_same}/{len(order)}",
           "first_step_logs": steps[:3], "last_step_logs": steps[-3:]}
    (out / "per_checkpoint_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("run_ok", "steps", "wall_time_s", "memory_per_step_summary", "selection_in_training", "rule_recheck",
                                          f"consistency_{G}_eval_vs_selection_answers_identical", f"consistency_init_sft_vs_S{seed}_eval_answers_identical")},
                     ensure_ascii=False, indent=1))
    for k, v in per_ck.items():
        print(k, v[KEY_METRICS[0]], v[KEY_METRICS[1]], v[KEY_METRICS[6]])
    return 0


def summary() -> int:
    out = V5 / "results" / "p42_p43"
    out.mkdir(parents=True, exist_ok=True)
    recs, order = load_data()
    summ = json.loads((AN / "summary.json").read_text(encoding="utf-8"))
    full = {g: {x["id"]: x for x in read_jsonl(AN / f"generations_{g}.jsonl")} for g in ALL}
    # 已有组的回答应与以往分析目录逐字相同（同一份生成）
    prev = {"S42": "hard_dev_s43", "S43": "hard_dev_s43", "C42": "hard_dev_s43", "B": "hard_dev_s43"}
    cons = {g: f'{sum(full[g][q]["answer"] == y for q, y in ((x["id"], x["answer"]) for x in read_jsonl(V5 / "results" / d / f"generations_{g}.jsonl")))}/{len(order)}'
            for g, d in prev.items()}
    rows = {g: [{"rec": recs[q], "s": score(recs[q], full[g][q]["answer"]), "answer": full[g][q]["answer"]} for q in order] for g in ALL}
    m = {g: metrics(rows[g]) for g in ALL}
    sty = {g: style(rows[g], full[g], recs) for g in ALL}
    for g in ALL:
        sty[g]["latency_ms_p50"] = summ["groups"][g]["metrics"]["cost"]["latency_ms_p50"]
    comps = {"P42_vs_S42": ("S42", "P42"), "P43_vs_S43": ("S43", "P43")}
    for p in ("P42", "P43"):
        for b in ("C42", "C43", "B", "A"):
            comps[f"{p}_vs_{b}"] = (b, p)
    pair = {k: paired(rows[a], rows[b]) for k, (a, b) in comps.items()}
    # 风格指标的配对差（逐题均值差；不做显著性门槛，只描述）
    def per_item_style(g):
        return [(len(clean(x["answer"])), copy_ratio(x["answer"], evidence_text(x["rec"]))) for x in rows[g]]
    style_diff = {}
    for k, (a, b) in comps.items():
        pa, pb = per_item_style(a), per_item_style(b)
        cr = [(y[1] - x[1]) for x, y in zip(pa, pb) if x[1] is not None and y[1] is not None]
        style_diff[k] = {"answer_chars_diff_mean": round(sum(y[0] - x[0] for x, y in zip(pa, pb)) / len(pa), 1),
                         "copy_ratio_diff_mean": round(sum(cr) / len(cr), 4)}
    gates = {p: gates_vs_b(rows, summ, p) for p in ("P42", "P43")}
    mean = lambda gs, k: round(sum(m[g][k]["rate"] for g in gs) / len(gs), 4) if all(m[g][k]["rate"] is not None for g in gs) else None
    seed_means = {k: {"P": mean(["P42", "P43"], k), "S": mean(["S42", "S43"], k), "C(v4)": mean(["C42", "C43"], k)} for k in KEY_METRICS}
    fa = {p: failure(rows[p], rows[s], s) for p, s in (("P42", "S42"), ("P43", "S43"))}
    keep = lambda r: r["id"] not in DISPUTED
    sens = {"note": "dv_m13、dv_m16 被人工抽查标为“标注有问题”；这里只报告剔除后的指标，不改变口径与检查点选择。",
            "full": {g: {k: fmt(m[g][k]) for k in KEY_METRICS[:2]} for g in ALL},
            "excluding": {g: {k: fmt(metrics(rows[g], keep)[k]) for k in KEY_METRICS[:2]} for g in ALL}}
    res = {"scorer": SCORER_VERSION, "status": "provisional（人工审核未完成的协议偏离保留；只基于 hard_dev 开发集）",
           "consistency_with_previous_analysis": cons, "groups": {g: {k: fmt(m[g][k]) for k in KEY_METRICS} for g in ALL},
           "rates": {g: {k: m[g][k]["rate"] for k in KEY_METRICS} for g in ALL}, "style": sty, "seed_means": seed_means,
           "paired": pair, "style_paired_diff": style_diff, "gates_dev_reference_only": gates, "failure": fa, "sensitivity_dv_m13_m16": sens,
           "copy_ratio_definition": f"去引用标记 / 引用行 / 空白后，回答中落在与证据逐字相同、长度 ≥ {COPY_MIN} 字符片段内的字符占比（逐题均值）"}
    (out / "dual_seed_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    # 表
    L = ["| 指标 | " + " | ".join(ALL) + " | P 均值 | S 均值 | v4 均值 |", "|" + "---|" * (len(ALL) + 4)]
    for k in KEY_METRICS:
        L.append(f"| {k} | " + " | ".join(fmt(m[g][k]) for g in ALL) + " | " +
                 " | ".join("—" if seed_means[k][x] is None else f"{seed_means[k][x]:.3f}" for x in ("P", "S", "C(v4)")) + " |")
    for sk in ("answer_chars_mean", "output_tokens_mean", "copy_ratio_mean", "copy_heavy_ge_0.5", "latency_ms_p50"):
        L.append(f"| {sk} | " + " | ".join(str(sty[g][sk]) for g in ALL) + " | | | |")
    L += ["", "| 配对（后者 − 前者） | 指标 | 差值 | 聚类 bootstrap 95% CI |", "|---|---|---|---|"]
    for key, v in pair.items():
        for k in [KEY_METRICS[i] for i in (0, 1, 3, 4, 6, 8, 10, 11)]:
            if k in v:
                L.append(f"| {key} | {k} | {v[k]['diff']:+.3f} | [{v[k]['ci95_cluster_bootstrap'][0]:+.3f}, {v[k]['ci95_cluster_bootstrap'][1]:+.3f}] |")
    L += ["", "| 配对 | 回答长度差（字符，均值） | 照抄比例差（均值） |", "|---|---|---|"]
    for k, v in style_diff.items():
        L.append(f"| {k} | {v['answer_chars_diff_mean']:+.1f} | {v['copy_ratio_diff_mean']:+.4f} |")
    (out / "dual_seed_table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(out / "per_item_dpo.csv", "w", encoding="utf-8-sig", newline="") as f:
        gs = ["A", "B", "C42", "C43", "S42", "S43", "P42", "P43"]
        cols = ["id", "behavior", "is_multi"] + [f"{g}_{x}" for g in gs for x in ("multi", "content")] + \
               ["P42_vs_S42", "P43_vs_S43", "P_agree_multi", "P_agree_content", "P42_missing", "P43_missing", "disputed_annotation"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(order):
            row = {"id": q, "behavior": recs[q]["behavior"], "is_multi": rows["P42"][i]["s"]["is_multi_v5"],
                   "disputed_annotation": "是" if q in DISPUTED else ""}
            for g in gs:
                row[f"{g}_multi"] = rows[g][i]["s"]["multi_complete_v5"]
                row[f"{g}_content"] = int(rows[g][i]["s"]["content_ok_v4"])
            for p, s in (("P42", "S42"), ("P43", "S43")):
                a, b = rows[s][i]["s"], rows[p][i]["s"]
                t = []
                if a["multi_complete_v5"] is not None and a["multi_complete_v5"] != b["multi_complete_v5"]:
                    t.append("多证据 " + ("↑" if b["multi_complete_v5"] else "↓"))
                if a["content_ok_v4"] != b["content_ok_v4"]:
                    t.append("内容 " + ("↑" if b["content_ok_v4"] else "↓"))
                row[f"{p}_vs_{s}"] = "；".join(t)
                row[f"{p}_missing"] = " / ".join(b["missing_facts"])
            row["P_agree_multi"] = row["P42_multi"] == row["P43_multi"]
            row["P_agree_content"] = row["P42_content"] == row["P43_content"]
            w.writerow(row)
    print("\n".join(L))
    print("consistency", cons)
    for p in gates:
        for k, v in gates[p].items():
            print(p, "GATE", k, v)
    print(json.dumps(fa, ensure_ascii=False)[:3000])
    print(json.dumps(sens["excluding"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if "--summary" in sys.argv:
        raise SystemExit(summary())
    raise SystemExit(single(next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seed="))))
