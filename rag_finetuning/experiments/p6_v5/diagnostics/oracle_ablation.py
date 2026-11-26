"""P6-v5 诊断：Oracle Evidence Ablation（只做推理，不改权重；只用 hard_dev，不用 test_v5）。

对象：hard_dev 中 43 道多证据题（scorer v4 is_multi_v5）。模型：A = 3B Base，C42 = 3B v4 种子 42，D = 7B Base。
条件（同模型、同题目、同解码参数）：
  A    原始 RAG 上下文（直接复用第一轮 hard_dev 输出：同一模型、同一冻结输入、同一解码，不重跑）
  B    仅 Gold Evidence：只保留金标准片段（按原证据中的先后顺序），删除全部干扰片段
  C    Gold Evidence + required_facts 清单（每条写出所在原句与片段 ID；诊断用，不代表真实能力）
  D    C + 生成后逐项事实覆盖检查：用评分器的事实匹配找出遗漏项；有遗漏时再调用一次模型做最小修订；
       修订若引入证据外数字或无效 ID 则退回 C 的答案（防止幻觉）
  Brev B 的证据顺序完全颠倒（只对 B 条件失败的题运行，检验顺序影响）

    python -m rag_finetuning.experiments.p6_v5.diagnostics.oracle_ablation check
    python -m rag_finetuning.experiments.p6_v5.diagnostics.oracle_ablation gen --model A --cond B
    python -m rag_finetuning.experiments.p6_v5.diagnostics.oracle_ablation analyze
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rag_finetuning.common import read_jsonl, write_jsonl
from rag_finetuning.evaluation.scoring import CITE_RE, normalize, numbers
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
DIAG = V5 / "diagnostics"
OUT = DIAG / "oracle_ablation"
REF = V5 / "data/hard_dev/built/hard_dev.jsonl"
ROUND1 = V5 / "results/hard_dev_round1"
RUNS = Path("workspace/runs/p6_v5/diagnostics/oracle_ablation")
MODELS = {"A": "A", "C42": "C42", "D": "D"}
MAXTOK = 384

CHECK_SUFFIX = ("\n\n【必要事实清单（诊断实验用，仅列出资料中已有的内容）】回答必须覆盖下列每一项，并在句末标注对应片段 ID：\n{items}")
VERIFY_USER = """问题：{q}

资料：
{ev}

当前回答：
{ans}

逐项核对发现，下列必要事实在资料中有明确依据，但当前回答没有写到：
{miss}

请输出修订后的完整回答：保留当前回答中正确的内容，补上遗漏的事实，每句话后用 [片段ID] 标注依据，最后一行写“引用：片段ID, 片段ID”。不要加入资料以外的内容。"""


def multi_items():
    rows = [r for r in read_jsonl(REF) if score_item_v4(r, r["reference_answer"])["is_multi_v5"]]
    return rows


def gold_evidence(r, reverse=False):
    gold = set(r["gold_cids"])
    ev = [e for e in r["evidence"] if e["cid"] in gold]
    return list(reversed(ev)) if reverse else ev


def fact_sentence(fact: str, text: str) -> str:
    alts = [normalize(a) for a in fact.split("|")]
    body = text.split("\n", 1)[-1]
    for s in re.split(r"(?<=[。；])", body):
        if any(a and a in normalize(s) for a in alts):
            return s.strip()
    return body.strip()


def checklist(r, ev):
    by = {e["cid"]: e["text"] for e in ev}
    items = []
    for f, c in zip(r["facts"], r["fact_cids"]):
        items.append(f"- {f.split('|')[0]}（原句：{fact_sentence(f, by[c])}）[{c}]")
    return "\n".join(items)


def cmd_check(a):
    """核查 Gold Evidence 是否足以支持全部必要事实：每条事实的出处片段在金标准中，且事实文本确实出现在该片段原文中。"""
    rows = multi_items()
    bad = []
    for r in rows:
        ev = {e["cid"]: e for e in gold_evidence(r)}
        for f, c in zip(r["facts"], r["fact_cids"]):
            if c not in ev or not any(normalize(x) in normalize(ev[c]["text"]) for x in f.split("|")):
                bad.append({"id": r["id"], "fact": f, "cid": c})
        s = score_item_v4(dict(r, evidence=gold_evidence(r)), r["reference_answer"])
        if not (s["content_ok_v4"] and s["multi_complete_v5"] and s["claim_support_full"]):
            bad.append({"id": r["id"], "fact": "参考答案在仅金标准证据下自检未通过", "cid": None})
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"n_multi_items": len(rows), "n_facts": sum(len(r["facts"]) for r in rows),
           "gold_sufficient_items": len(rows) - len({b["id"] for b in bad}), "problems": bad,
           "gold_chunks_per_item": dict(Counter(len(r["gold_cids"]) for r in rows)),
           "distractors_removed_per_item_mean": round(statistics.mean(len(r["evidence"]) - len(r["gold_cids"]) for r in rows), 2)}
    (OUT / "gold_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, ensure_ascii=False), "problems:", bad)


def cmd_gen(a):
    from rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 import GROUPS, make_chat
    from rag_finetuning.compat.rag_pipeline import build_messages
    rows = multi_items()
    if a.cond == "Brev":
        failed = {x["id"] for x in read_jsonl(RUNS / a.model / "B" / "generations.jsonl")
                  if not score_item_v4(next(r for r in rows if r["id"] == x["id"]) | {"evidence": gold_evidence(next(r for r in rows if r["id"] == x["id"]))},
                                       x["answer"])["multi_complete_v5"]}
        rows = [r for r in rows if r["id"] in failed]
    out = RUNS / a.model / a.cond
    if (out / "generations.jsonl").exists():
        raise SystemExit(f"{out} 已有结果，不覆盖")
    chat, settings, name, oll = make_chat(GROUPS[MODELS[a.model]])
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        ev = gold_evidence(r, reverse=(a.cond == "Brev"))
        m = build_messages(r["question"], [(e["cid"], e["text"]) for e in ev], {e["cid"]: e["source"] for e in ev})
        if a.cond in ("C", "D"):
            m[1]["content"] += CHECK_SUFFIX.format(items=checklist(r, ev))
        x = chat(m, MAXTOK)
        rec = {"id": r["id"], "model": a.model, "cond": a.cond, "answer": x["text"], "calls": [{"stage": "generate", **{k: x[k] for k in ("latency_ms", "output_tokens", "input_tokens")}}],
               "evidence_cids": [e["cid"] for e in ev]}
        if a.cond == "D":
            rr = dict(r, evidence=ev)
            s = score_item_v4(rr, x["text"])
            miss = [(d["fact"], d["cid"]) for d in s["fact_detail"] if not d["hit"]]
            rec["missing_before"] = [f for f, _ in miss]
            rec["revised"] = False
            if miss:
                by = {e["cid"]: e["text"] for e in ev}
                y = chat([{"role": "user", "content": VERIFY_USER.format(q=r["question"], ev="\n\n".join(f"[{e['cid']}] {e['text']}" for e in ev),
                                                                          ans=x["text"], miss="\n".join(f"- {fact_sentence(f, by[c])} [{c}]" for f, c in miss))}], MAXTOK + 64)
                rec["calls"].append({"stage": "verify_revise", **{k: y[k] for k in ("latency_ms", "output_tokens", "input_tokens")}})
                ev_nums = set(numbers(" ".join(e["text"] for e in ev))) | set(numbers(r["question"]))
                bad_ids = set(CITE_RE.findall(y["text"])) - set(by)
                bad_nums = set(numbers(re.split(r"\n?\s*引用\s*[：:]", y["text"], maxsplit=1)[0])) - ev_nums
                rec["revision_raw"] = y["text"]
                rec["revision_guard"] = {"bad_ids": sorted(bad_ids), "bad_nums": sorted(bad_nums)}
                if y["text"].strip() and not bad_ids and not bad_nums:
                    rec["answer"], rec["revised"] = y["text"], True
        rec["latency_ms"] = round(sum(c["latency_ms"] for c in rec["calls"]), 1)
        rec["output_tokens"] = sum(c["output_tokens"] for c in rec["calls"])
        rec["truncated"] = any(c["output_tokens"] >= MAXTOK for c in rec["calls"])
        res.append(rec)
        print(f"[{i}/{len(rows)}] {r['id']} {rec['latency_ms']:.0f}ms", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "generations.jsonl", res)
    (out / "meta.json").write_text(json.dumps({"model": name, "cond": a.cond, "settings": settings, "n": len(res), "wall_s": round(time.time() - t0, 1)},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    if oll is not None:
        import os
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(os.environ.get("P6_OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate", method="POST",
                                                      data=json.dumps({"model": "qwen2.5:7b", "keep_alive": 0}).encode()), timeout=60).read()


DECL = re.compile(r"(没有(说明|提到|提及|规定)|未(说明|提及|提到)|无法(回答|确定)|不清楚)")


def attribute(r, s, ans, truncated):
    """单题错误归因（按优先级取一个主因；条件 A 用原始证据，其余用金标准证据评分）。"""
    if s["content_ok_v4"] and s["multi_complete_v5"]:
        if s["citation_accuracy"] is False or not s["citation_valid"]:
            return "引用错误（内容完整）"
        return "正确"
    if truncated:
        return "输出截断"
    if s["unsupported_numbers_v4"] or s["forbidden_hits"]:
        return "无依据推断"
    if s["whole_refusal_on_answerable"] or (s["missing_facts"] and DECL.search(ans or "")):
        return "证据不足（模型声称资料没有说明）"
    if not s["missing_facts"] and not s["content_ok_v4"]:
        return "条件误解（事实都写到但结论或行为错误）"
    if r.get("no_conflict") and s.get("conflict_claimed"):
        return "条件误解（把范围 / 条件 / 版本差异判为冲突）"
    if s["missing_facts"]:
        return "事实遗漏"
    return "其他"


def cmd_analyze(a):
    rows = {r["id"]: r for r in multi_items()}
    r1 = {}
    for m in MODELS:
        r1[m] = {x["id"]: x for x in read_jsonl(ROUND1 / f"generations_{m}.jsonl")}
    data = defaultdict(dict)       # data[model][cond][id] = (answer, rec)
    for m in MODELS:
        data[m]["A"] = {q: {"answer": r1[m][q]["answer"], "latency_ms": r1[m][q]["latency_ms"], "output_tokens": r1[m][q]["output_tokens"],
                            "truncated": (r1[m][q]["output_tokens"] or 0) >= MAXTOK} for q in rows}
        for c in ("B", "C", "D", "Brev"):
            p = RUNS / m / c / "generations.jsonl"
            if p.exists():
                data[m][c] = {x["id"]: x for x in read_jsonl(p)}
    scores = defaultdict(dict)
    per = []
    for m in MODELS:
        for c, g in data[m].items():
            for q, x in g.items():
                r = rows[q]
                rr = r if c == "A" else dict(r, evidence=[e for e in r["evidence"] if e["cid"] in set(x.get("evidence_cids") or r["gold_cids"])])
                if c == "Brev":
                    rr = dict(r, evidence=list(reversed([e for e in r["evidence"] if e["cid"] in set(r["gold_cids"])])))
                s = score_item_v4(rr, x["answer"])
                scores[(m, c)][q] = s
                per.append({"model": m, "cond": c, "id": q, "multi_complete": int(bool(s["multi_complete_v5"])), "content_ok": int(s["content_ok_v4"]),
                            "fact_cov": round(s["required_fact_coverage"] or 0, 3), "missing": " / ".join(s["missing_facts"]),
                            "cause": attribute(r, s, x["answer"], x.get("truncated")), "revised": x.get("revised"),
                            "output_tokens": x.get("output_tokens"), "latency_ms": x.get("latency_ms"), "answer": x["answer"]})
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"n": len(rows), "note": "自动规则评分，provisional；C / D 条件提供了必要事实清单，仅用于诊断，不代表真实能力", "by_model_cond": {}, "transitions": {},
            "causes": {}, "verify": {}, "order": {}}
    for (m, c), S in scores.items():
        g = data[m][c]
        n = len(S)
        summ["by_model_cond"][f"{m}|{c}"] = {
            "n": n, "multi_complete": sum(bool(s["multi_complete_v5"]) for s in S.values()), "content_ok": sum(s["content_ok_v4"] for s in S.values()),
            "fact_coverage_mean": round(statistics.mean(s["required_fact_coverage"] or 0 for s in S.values()), 3),
            "claim_support_full": sum(bool(s["claim_support_full"]) for s in S.values()),
            "unsupported_number_items": sum(bool(s["unsupported_numbers_v4"]) for s in S.values()),
            "output_tokens_mean": round(statistics.mean(x.get("output_tokens") or 0 for x in g.values()), 1),
            "latency_ms_p50": statistics.median(x.get("latency_ms") or 0 for x in g.values()),
            "truncated": sum(bool(x.get("truncated")) for x in g.values())}
        summ["causes"][f"{m}|{c}"] = dict(Counter(p["cause"] for p in per if p["model"] == m and p["cond"] == c))
    for m in MODELS:
        ok = {c: {q for q, s in scores[(m, c)].items() if s["multi_complete_v5"]} for c in ("A", "B", "C", "D") if (m, c) in scores}
        if all(c in ok for c in "ABCD"):
            summ["transitions"][m] = {
                "A_fail": len(rows) - len(ok["A"]),
                "fixed_by_gold(A失败→B成功)": len((set(rows) - ok["A"]) & ok["B"]), "broken_by_gold(A成功→B失败)": len(ok["A"] - ok["B"]),
                "fixed_by_checklist(B失败→C成功)": len((set(rows) - ok["B"]) & ok["C"]), "broken_by_checklist": len(ok["B"] - ok["C"]),
                "fixed_by_verify(C失败→D成功)": len((set(rows) - ok["C"]) & ok["D"]), "broken_by_verify": len(ok["C"] - ok["D"]),
                "still_fail_in_D": sorted(set(rows) - ok["D"])}
            sC, sD = scores[(m, "C")], scores[(m, "D")]
            rev = [q for q, x in data[m]["D"].items() if x.get("revised")]
            summ["verify"][m] = {"items_with_missing_after_C": sum(1 for x in data[m]["D"].values() if x.get("missing_before")),
                                 "revisions_accepted": len(rev),
                                 "revisions_rejected_by_guard": sum(1 for x in data[m]["D"].values() if x.get("missing_before") and not x.get("revised")),
                                 "new_unsupported_numbers_after_verify": sum(1 for q in rows if sD[q]["unsupported_numbers_v4"] and not sC[q]["unsupported_numbers_v4"]),
                                 "claim_support_lost_after_verify": sum(1 for q in rows if sC[q]["claim_support_full"] and not sD[q]["claim_support_full"])}
        if (m, "Brev") in scores:
            br = scores[(m, "Brev")]
            summ["order"][m] = {"B_failed_items": len(br), "fixed_by_reversing_order": sum(bool(s["multi_complete_v5"]) for s in br.values()),
                                "items": {q: bool(s["multi_complete_v5"]) for q, s in br.items()}}
    (OUT / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "per_item_per_condition.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(sorted(per, key=lambda p: (p["id"], p["model"], "A B C D Brev".split().index(p["cond"]))))
    # 逐题对比表：每题一行，四种条件的完整回答与差异
    with open(OUT / "per_item_diff.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["id", "question", "required_facts", "reference_answer"]
        for m in MODELS:
            for c in ("A", "B", "C", "D"):
                cols += [f"{m}_{c}_multi", f"{m}_{c}_missing", f"{m}_{c}_cause", f"{m}_{c}_answer"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        idx = {(p["model"], p["cond"], p["id"]): p for p in per}
        for q, r in rows.items():
            row = {"id": q, "question": r["question"], "required_facts": " / ".join(r["facts"]), "reference_answer": r["reference_answer"]}
            for m in MODELS:
                for c in ("A", "B", "C", "D"):
                    p = idx.get((m, c, q))
                    if p:
                        row.update({f"{m}_{c}_multi": p["multi_complete"], f"{m}_{c}_missing": p["missing"], f"{m}_{c}_cause": p["cause"], f"{m}_{c}_answer": p["answer"]})
            w.writerow(row)
    print(json.dumps({k: summ[k] for k in ("by_model_cond", "transitions", "verify", "order")}, ensure_ascii=False, indent=1))
    print(json.dumps(summ["causes"], ensure_ascii=False, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True, choices=list(MODELS))
    g.add_argument("--cond", required=True, choices=["B", "C", "D", "Brev"])
    sub.add_parser("analyze")
    a = ap.parse_args(argv)
    {"check": cmd_check, "gen": cmd_gen, "analyze": cmd_analyze}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
