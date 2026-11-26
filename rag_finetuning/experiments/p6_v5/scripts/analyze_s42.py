"""S42 / S43 正式 SFT 的训练后分析（只读；不重新生成、不改选择结果）。--seed=43 分析 S43（输入 hard_dev_s43、输出 results/s43），默认 42。

    python -m rag_finetuning.experiments.p6_v5.scripts.analyze_s42

输入：
  runs/p6_v5/sft_3b_s42/{run_record.json, train_log.jsonl, dev_eval/checkpoint_epoch*.jsonl}
  results/hard_dev_s42/（run_eval_v5 analyze：A / B / C42 / C43 / D / S42 等组，评分器 4.0.3）
输出 results/s42/：
  training_curve.svg、train_log.csv、per_epoch_metrics.json、comparison.json、per_item_changes.csv、
  sensitivity_dv_m13_m16.json、failure_analysis.json
所有结果为自动评分（provisional）；dv_m13 / dv_m16 敏感性分析只报告，不改变检查点选择。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as sv2
from rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 import METRICS, mean_rate
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4

V5 = Path(__file__).resolve().parents[1]
SEED = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--seed=")), "42")   # 2026-11-24 加入：--seed=43 分析 S43，默认 42 与原行为相同
G = f"S{SEED}"
RUN = Path(f"workspace/runs/p6_v5/sft_3b_s{SEED}")
AN = V5 / "results" / f"hard_dev_s{SEED}"
OUT = V5 / "results" / f"s{SEED}"
GROUPS = ["A", "B", "C42", "C43", "D"]
DISPUTED = {"dv_m13", "dv_m16"}
KEY_METRICS = ["multi_complete_v5（多证据完整率，主指标）", "content_ok_v4（内容正确，不含引用格式）", "required_fact_coverage（必要事实覆盖率，均值）",
               "claim_support_full（陈述支持，全分母）", "citation_accuracy（事实引用正确）", "citation_valid（引用 ID 合法）",
               "whole_refusal（应回答类整体拒答）", "false_refusal（answer 题错误拒答）", "refusal_ok（正确拒答）", "partial_ok",
               "conflict_true_ok（真冲突识别）", "no_conflict_ok（非冲突判断）", "injection_resisted", "format_ok",
               "first_evidence_only（多证据题只用第一个）"]


def score(rec, ans):
    s = score_item_v4(rec, ans)
    s["first_evidence_only_m"] = s["first_evidence_only"] if s["is_multi_v5"] else None
    return s


def metrics(rows, keep=lambda r: True):
    return {k: mean_rate(x["s"][key] for x in rows if sel(x["rec"]) and keep(x["rec"])) for k, (sel, key, _) in METRICS.items()}


def fmt(m):
    return "—" if not m["n"] else (f"{m['rate']:.2f}" if isinstance(m["k"], float) else f"{m['k']}/{m['n']}")


def svg_curve(log, path):
    steps = [x for x in log if "loss" in x]
    evals = [x for x in log if "eval_loss" in x]
    W, H, P = 860, 210, 46
    panels = [("训练 loss（每步）与 hard_dev loss（每 epoch，橙点）", [x["loss"] for x in steps], [(x["step"], x["eval_loss"]) for x in evals]),
              ("学习率", [x["learning_rate"] for x in steps], []), ("梯度范数（裁剪前）", [x["grad_norm"] for x in steps], [])]
    n = len(steps)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H * 3}" font-family="Microsoft YaHei, sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="#fff"/>']
    for i, (title, ys, pts) in enumerate(panels):
        y0 = i * H
        allv = ys + [p[1] for p in pts]
        lo, hi = min(allv), max(allv)
        hi = hi if hi > lo else lo + 1
        X = lambda s: P + (s - 1) / max(1, n - 1) * (W - 2 * P)
        Y = lambda v: y0 + H - 30 - (v - lo) / (hi - lo) * (H - 60)
        out.append(f'<text x="{P}" y="{y0 + 18}" font-weight="bold">{title}</text>')
        out.append(f'<line x1="{P}" y1="{y0 + H - 30}" x2="{W - P}" y2="{y0 + H - 30}" stroke="#bbb"/>')
        for v in (lo, (lo + hi) / 2, hi):
            out.append(f'<text x="4" y="{Y(v) + 4:.1f}" fill="#666">{v:.3g}</text>')
        for e in range(1, 4):
            s = e * n / 3
            out.append(f'<line x1="{X(s):.1f}" y1="{y0 + 26}" x2="{X(s):.1f}" y2="{y0 + H - 30}" stroke="#eee"/>'
                       f'<text x="{X(s) - 30:.1f}" y="{y0 + H - 14}" fill="#666">epoch {e}（step {round(s)}）</text>')
        out.append('<polyline fill="none" stroke="#2c5cad" stroke-width="1.3" points="' +
                   " ".join(f"{X(k + 1):.1f},{Y(v):.1f}" for k, v in enumerate(ys)) + '"/>')
        for s, v in pts:
            out.append(f'<circle cx="{X(max(1, s)):.1f}" cy="{Y(v):.1f}" r="4" fill="#e07b00"/><text x="{X(max(1, s)) + 6:.1f}" y="{Y(v) - 6:.1f}" fill="#e07b00">{v:.3f}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rr = json.loads((RUN / "run_record.json").read_text(encoding="utf-8"))
    log = [json.loads(x) for x in (RUN / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    svg_curve(log, OUT / "training_curve.svg")
    with open(OUT / "train_log.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "loss", "learning_rate", "grad_norm", "eval_loss", "mem_alloc_peak_mib", "mem_reserved_peak_mib", "t_s"])
        for x in log:
            w.writerow([x.get("step"), x.get("epoch"), x.get("loss"), x.get("learning_rate"), x.get("grad_norm"), x.get("eval_loss"),
                        x.get("mem_alloc_peak_mib"), x.get("mem_reserved_peak_mib"), x.get("t")])
    recs = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    order = [r["id"] for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")]
    # ---- 各检查点（训练内选择时保存的逐题回答，评分器 4.0.3 重新计算全部指标）
    sel = rr["epoch_selection"]
    per_epoch, ep_rows = {}, {}
    for ck in sorted(RUN.glob("dev_eval/checkpoint_epoch*.jsonl")):
        e = int(ck.stem.replace("checkpoint_epoch", ""))
        ans = {x["id"]: x["answer"] for x in read_jsonl(ck)}
        rows = [{"rec": recs[q], "s": score(recs[q], ans[q]), "answer": ans[q]} for q in order]
        ep_rows[e] = rows
        per_epoch[e] = {k: metrics(rows)[k] for k in KEY_METRICS}
    rule_check = sorted(sel["per_epoch"], key=lambda x: (-x["dev_multi_v5"], -x["dev_content_v4"], x["dev_whole_refusal"], x["epoch"]))[0]["epoch"]
    (OUT / "per_epoch_metrics.json").write_text(json.dumps({"scorer": SCORER_VERSION, "selection_in_training": sel, "rule_recheck_epoch": rule_check,
                                                            "per_epoch_full_metrics": per_epoch}, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- 组间对照（run_eval_v5 analyze 的结果）
    summ = json.loads((AN / "summary.json").read_text(encoding="utf-8"))
    gens = {g: {x["id"]: x["answer"] for x in read_jsonl(AN / f"generations_{g}.jsonl")} for g in GROUPS + [G] if (AN / f"generations_{g}.jsonl").exists()}
    grows = {g: [{"rec": recs[q], "s": score(recs[q], gens[g][q]), "answer": gens[g][q]} for q in order] for g in gens}
    sel_e = sel["selected_epoch"]
    same = sum(grows[G][i]["answer"] == ep_rows[sel_e][i]["answer"] for i in range(len(order)))
    cmp = {"groups": {g: {k: summ["groups"][g]["metrics"][k] for k in KEY_METRICS} | {"cost": summ["groups"][g]["metrics"]["cost"]} for g in gens},
           "paired_vs": {}, "selected_epoch": sel_e,
           f"consistency_{G}_eval_vs_selection_answers_identical": f"{same}/{len(order)}"}
    for base in ("B", "C42", "C43", "A") if SEED != "42" else ("B", "C42", "A"):   # S42 已提交结果不含 C43 配对，保持不变
        cmp["paired_vs"][f"{base}_vs_{G}"] = {k: sv2.paired_cluster([{"rec": x["rec"], "s": {"v": x["s"][key]}} for x in grows[base]],
                                                                    [{"rec": x["rec"], "s": {"v": x["s"][key]}} for x in grows[G]], "v", sl,
                                                                    lower_is_better=low)
                                              for k, (sl, key, low) in METRICS.items() if k in KEY_METRICS and key != "required_fact_coverage"}
    # 预注册晋级门槛（开发集参考：只用于描述，不作晋级判定；正式判定只在 test_v5）
    P = cmp["paired_vs"][f"B_vs_{G}"]
    best_content = max(("B", "C42", "C43"), key=lambda g: summ["groups"][g]["metrics"][KEY_METRICS[1]]["rate"])
    pc = sv2.paired_cluster([{"rec": x["rec"], "s": {"v": x["s"]["content_ok_v4"]}} for x in grows[best_content]],
                            [{"rec": x["rec"], "s": {"v": x["s"]["content_ok_v4"]}} for x in grows[G]], "v")
    g = lambda k: P[k]
    unsup = {x: sum(bool(r["s"]["unsupported_numbers_v4"]) for r in grows[x]) for x in ("B", G)}
    lat = {x: summ["groups"][x]["metrics"]["cost"]["latency_ms_p50"] for x in ("B", G)}
    gates = {
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
    cmp["gates_dev_reference_only"] = gates
    (OUT / "comparison.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- 逐题变化
    with open(OUT / "per_item_changes.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["id", "behavior", "subtype", "is_multi"] + [f"{x}_{m}" for x in ["A", "B", "C42"] + [f"{G}_e{e}" for e in sorted(ep_rows)] + [G]
                                                         for m in ("multi", "content", "missing")] + ["change_vs_C42", "change_vs_B", "disputed_annotation"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(order):
            r = recs[q]
            row = {"id": q, "behavior": r["behavior"], "subtype": r.get("subtype") or "", "is_multi": grows[G][i]["s"]["is_multi_v5"],
                   "disputed_annotation": "是（人工抽查标为标注有问题）" if q in DISPUTED else ""}
            srcs = {x: grows[x][i] for x in ("A", "B", "C42", G)} | {f"{G}_e{e}": ep_rows[e][i] for e in ep_rows}
            for x, z in srcs.items():
                row.update({f"{x}_multi": z["s"]["multi_complete_v5"], f"{x}_content": int(z["s"]["content_ok_v4"]),
                            f"{x}_missing": " / ".join(z["s"]["missing_facts"])})

            def ch(base):
                a, b = srcs[base]["s"], srcs[G]["s"]
                t = []
                if a["multi_complete_v5"] is not None and a["multi_complete_v5"] != b["multi_complete_v5"]:
                    t.append("多证据 " + ("↑" if b["multi_complete_v5"] else "↓"))
                if a["content_ok_v4"] != b["content_ok_v4"]:
                    t.append("内容 " + ("↑" if b["content_ok_v4"] else "↓"))
                return "；".join(t)
            row["change_vs_C42"], row["change_vs_B"] = ch("C42"), ch("B")
            w.writerow(row)
    # ---- dv_m13 / dv_m16 敏感性（只报告）
    keep = lambda r: r["id"] not in DISPUTED
    sens = {"note": "人工抽查把 dv_m13、dv_m16 标为“标注有问题”。这里只报告剔除这两题后的指标与检查点排序，不改变正式口径与检查点选择。",
            "groups_full": {x: {k: fmt(metrics(grows[x])[k]) for k in KEY_METRICS[:2]} for x in grows},
            "groups_excluding": {x: {k: fmt(metrics(grows[x], keep)[k]) for k in KEY_METRICS[:2]} for x in grows},
            "epochs_full": {e: {k: fmt(metrics(ep_rows[e])[k]) for k in KEY_METRICS[:2]} for e in ep_rows},
            "epochs_excluding": {e: {k: fmt(metrics(ep_rows[e], keep)[k]) for k in KEY_METRICS[:2]} for e in ep_rows},
            "per_item": {q: {x: {"multi": grows[x][order.index(q)]["s"]["multi_complete_v5"], "content": grows[x][order.index(q)]["s"]["content_ok_v4"],
                                 "missing": grows[x][order.index(q)]["s"]["missing_facts"]} for x in grows} for q in sorted(DISPUTED)}}
    m_ex = {e: metrics(ep_rows[e], keep) for e in ep_rows}
    sens["selection_if_excluded（仅供参考，未采用）"] = sorted(ep_rows, key=lambda e: (-m_ex[e][KEY_METRICS[0]]["k"], -m_ex[e][KEY_METRICS[1]]["k"],
                                                                         m_ex[e][KEY_METRICS[6]]["k"], e))[0]
    sens["selection_official"] = sel_e
    (OUT / "sensitivity_dv_m13_m16.json").write_text(json.dumps(sens, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- 失败分析（选中检查点）
    S = grows[G]
    fail_multi = [x for x in S if x["s"]["multi_complete_v5"] is False]
    fa = {"multi_fail_n": len(fail_multi), "multi_n": sum(x["s"]["is_multi_v5"] for x in S),
          "multi_fail_pattern": dict(Counter("只用第一个金标准片段" if x["s"]["first_evidence_only"] else "只用最后一个金标准片段" if x["s"]["last_evidence_only"]
                                             else ("整体拒答" if x["s"]["whole_refusal_on_answerable"] else "部分事实缺失") for x in fail_multi)),
          "missing_fact_counts": dict(Counter(f for x in fail_multi for f in x["s"]["missing_facts"]).most_common(15)),
          "content_fail_by_behavior": dict(Counter(x["rec"]["behavior"] for x in S if not x["s"]["content_ok_v4"])),
          "content_fail_reasons": dict(Counter(("证据外数字" if x["s"]["unsupported_numbers_v4"] else "") or ("禁止内容" if x["s"]["forbidden_hits"] else "") or
                                               ("缺失必要事实" if x["s"]["missing_facts"] else "") or "行为错误（拒答 / 冲突 / 部分回答判断）"
                                               for x in S if not x["s"]["content_ok_v4"])),
          "lost_vs_C42_multi": [x["rec"]["id"] for x, c in zip(S, grows["C42"]) if c["s"]["multi_complete_v5"] and x["s"]["multi_complete_v5"] is False],
          "gained_vs_C42_multi": [x["rec"]["id"] for x, c in zip(S, grows["C42"]) if c["s"]["multi_complete_v5"] is False and x["s"]["multi_complete_v5"]],
          "lost_vs_C42_content": [x["rec"]["id"] for x, c in zip(S, grows["C42"]) if c["s"]["content_ok_v4"] and not x["s"]["content_ok_v4"]],
          "gained_vs_C42_content": [x["rec"]["id"] for x, c in zip(S, grows["C42"]) if not c["s"]["content_ok_v4"] and x["s"]["content_ok_v4"]],
          "unsupported_numbers_items": {x["rec"]["id"]: x["s"]["unsupported_numbers_v4"] for x in S if x["s"]["unsupported_numbers_v4"]},
          "output_tokens_mean": summ["groups"][G]["metrics"]["cost"]["output_tokens_mean"]}
    (OUT / "failure_analysis.json").write_text(json.dumps(fa, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- 控制台摘要
    print("selected_epoch", sel_e, "rule_recheck", rule_check, G, "gen vs selection answers identical", same, "/", len(order))
    cols = ["A", "B", "C42", "C43", "D"] + [f"{G}_e{e}" for e in sorted(ep_rows)] + [G]
    allm = {x: metrics(grows[x]) for x in grows} | {f"{G}_e{e}": metrics(ep_rows[e]) for e in ep_rows}
    print("| 指标 | " + " | ".join(cols) + " |")
    for k in KEY_METRICS:
        print(f"| {k} | " + " | ".join(fmt(allm[c][k]) for c in cols if c in allm) + " |")
    for k, v in gates.items():
        print("GATE", k, v)
    print(json.dumps(fa, ensure_ascii=False)[:1500])
    print(json.dumps({k: sens[k] for k in ("groups_full", "groups_excluding", "epochs_full", "epochs_excluding", "selection_if_excluded（仅供参考，未采用）")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
