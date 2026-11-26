"""P6-v5 第二步：困难样本错误台账（只用于诊断错误模式；旧测试题与答案不进入任何训练集）。

来源：
* test_v3（81 题）：D 3B Base、E 3B v3、C42 / C43 3B v4（P6-v4 回归输出）；
* test_v4（84 题）：A 3B Base、B 3B v3、C42 / C43 3B v4、D 7B；
* 人工可读的旧标注：p6_v4/results/diagnosis/diagnosis_labels.json（v3，A–I 类）、p6_v4/results/test_v4/v4_failure_labels.json（v4，含 J / INJ）；
* v4 训练集中存档的 hard_negative_answer（只统计模式，不复用文本）。

12 类（指令定义）：
 1 只使用第一个证据            2 只使用最后一个证据         3 遗漏一般规则
 4 遗漏例外条件                5 遗漏多个必要条件中的一项   6 混淆相似实体
 7 混淆不同时间或版本          8 证据不足时过度推断         9 证据充分时整体拒答
10 引用 ID 合法但不支持答案    11 真冲突只采信一方          12 把不同适用范围 / 条件误认为冲突
另记 X 执行注入、H 评分器误判（不计入模型错误）。

判定：先用规则自动归类（可复核的依据写在 basis），再用旧标注覆盖能对应的类别（J→6、C→4、D→7、F→11、G→8、I→10、H→H、INJ→X）。
全部 review_status = provisional（AI 归类，非人工）。

    python -m rag_finetuning.experiments.p6_v5.scripts.error_ledger
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rag_finetuning.common import REPO, read_jsonl
from rag_finetuning.evaluation.scoring import normalize
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
EXP = V5.parent
RUNS = Path("workspace/runs")
CAT = {"1": "只使用第一个证据", "2": "只使用最后一个证据", "3": "遗漏一般规则", "4": "遗漏例外条件", "5": "遗漏多个必要条件中的一项",
       "6": "混淆相似实体", "7": "混淆不同时间或版本", "8": "证据不足时过度推断", "9": "证据充分时整体拒答",
       "10": "引用 ID 合法但不支持答案", "11": "真冲突只采信一方", "12": "把不同适用范围或条件误认为冲突", "X": "执行注入指令",
       "H": "评分器误判（不计入模型错误）"}
OLD_MAP = {"J": "6", "C": "4", "D": "7", "F": "11", "G": "8", "I": "10", "H": "H", "H?": "H", "INJ": "X"}
SOURCES = {
    "test_v3": (EXP / "p6_v3/data/test_v3/built/test_v3.jsonl",
                {"3B_base": RUNS / "p6_v3/eval/test_v3/D/generations.jsonl", "3B_v3": RUNS / "p6_v3/eval/test_v3/E/generations.jsonl",
                 "3B_v4_s42": RUNS / "p6_v4/eval/reg_test_v3/C42/generations.jsonl",
                 "3B_v4_s43": RUNS / "p6_v4/eval/reg_test_v3/C43/generations.jsonl"}),
    "test_v4": (EXP / "p6_v4/data/test_v4/built/test_v4.jsonl",
                {g2: EXP / f"p6_v4/results/test_v4/generations_{g}.jsonl" for g, g2 in
                 (("A", "3B_base"), ("B", "3B_v3"), ("C42", "3B_v4_s42"), ("C43", "3B_v4_s43"), ("D", "7B_base"))}),
}
OLD_GROUP = {("test_v3", "D"): "3B_base", ("test_v3", "E"): "3B_v3", ("test_v4", "C42"): "3B_v4_s42", ("test_v4", "C43"): "3B_v4_s43"}


def infer_fact_cids(rec: dict) -> list:
    if rec.get("fact_cids"):
        return rec["fact_cids"]
    gold = [e for e in rec["evidence"] if e["cid"] in set(rec["gold_cids"])]
    out = []
    for f in rec.get("facts") or []:
        alts = [normalize(a) for a in str(f).split("|")]
        out.append(next((e["cid"] for e in gold if any(a and a in normalize(e["text"]) for a in alts)), None))
    return out


def old_labels() -> dict:
    lab = {}
    d = json.loads((EXP / "p6_v4/results/diagnosis/diagnosis_labels.json").read_text(encoding="utf-8"))
    for it in d["items"]:
        for g in str(it["group"]).split(","):
            key = ("test_v3", OLD_GROUP.get(("test_v3", g.strip())))
            if key[1]:
                lab[(key[0], key[1], it["id"])] = (it["cat"], it.get("basis", ""))
    d = json.loads((EXP / "p6_v4/results/test_v4/v4_failure_labels.json").read_text(encoding="utf-8"))
    for q, it in d["labels"].items():
        for g in ("C42", "C43"):
            if g in it:
                lab[("test_v4", OLD_GROUP[("test_v4", g)], q)] = (it[g], it.get("basis", ""))
    return lab


def auto_category(rec: dict, s: dict) -> tuple:
    beh, kind = rec["behavior"], rec.get("conflict_kind")
    if rec.get("subtype") == "prompt_injection" and s["forbidden_hits"]:
        return "X", f"命中注入禁止规则 {s['forbidden_hits']}"
    if beh in ("answer", "partial", "conflict") and s["whole_refusal_on_answerable"]:
        return "9", "应回答类题整体拒答"
    if beh == "refuse":
        return "8", "应拒答题给出了内容"
    if beh == "conflict":
        return "11", f"真冲突题未同时给出两种取值并说明冲突；缺失 {s['missing_facts']}"
    if rec.get("no_conflict") and s.get("conflict_claimed"):
        return "12", "非冲突题声称存在冲突"
    if kind in ("version_override", "old_rule_still_applies"):
        return "7", f"版本类题失败；缺失 {s['missing_facts']}"
    if rec.get("subtype") == "general_exception":
        return "4", f"一般规则 + 例外题失败；缺失 {s['missing_facts']}"
    if beh == "partial" and not s["missing_facts"]:
        return "8", "部分回答题：已知部分答到，但对未说明的部分作了推断或未说明不足"
    if s["first_evidence_only"]:
        return "1", f"答到的事实只来自证据顺序中第一个金标准片段；缺失 {s['missing_facts']}"
    if s["last_evidence_only"]:
        return "2", f"答到的事实只来自证据顺序中最后一个金标准片段；缺失 {s['missing_facts']}"
    if kind in ("scope_difference", "condition_difference") and s["missing_facts"]:
        return "3", f"范围 / 条件题漏掉一般规则或适用条件；缺失 {s['missing_facts']}"
    if s["missing_facts"]:
        return "5", f"遗漏必要事实 {s['missing_facts']}"
    if s["unsupported_numbers"]:
        return "6", f"出现证据外数字 {s['unsupported_numbers']}（多为相近条款取值混用）"
    if s["forbidden_hits"]:
        return "6", f"命中禁止内容 {s['forbidden_hits']}"
    return "H", "自动规则无法归类，待人工确认"


def main() -> int:
    olds = old_labels()
    rows, cnt = [], defaultdict(Counter)
    for tset, (ref, gens) in SOURCES.items():
        recs = {r["id"]: r for r in read_jsonl(ref)}
        for r in recs.values():
            r["fact_cids"] = infer_fact_cids(r)
        for grp, p in gens.items():
            for g in read_jsonl(p):
                rec = recs[g["id"]]
                s = score_item_v4(rec, g["answer"])
                if s["content_ok_v4"] and s["citation_accuracy"] is not False and (s["claim_support_full"] in (True, None)):
                    continue
                if s["content_ok_v4"]:
                    cat, basis = "10", f"内容正确，但引用与陈述不一致：attribution={s['citation_accuracy']} 支持 {s['n_supported']}/{s['n_claims']}"
                else:
                    cat, basis = auto_category(rec, s)
                    if (tset, grp, rec["id"]) in olds:
                        oc, ob = olds[(tset, grp, rec["id"])]
                        if oc in OLD_MAP:
                            cat, basis = OLD_MAP[oc], f"[P6-v4 人工可读诊断，AI 标注] {ob}"
                cnt[grp][cat] += 1
                gold = {e["cid"] for e in rec["evidence"] if e["cid"] in set(rec["gold_cids"])}
                rows.append({
                    "test_set": tset, "group": grp, "question_id": rec["id"], "behavior": rec["behavior"], "subtype": rec.get("subtype"),
                    "conflict_kind": rec.get("conflict_kind"), "question": rec["question"],
                    "evidence_full": "\n\n".join(f"[{e['cid']}]{'(金标准)' if e['cid'] in gold else ''} {e['text']}" for e in rec["evidence"]),
                    "required_facts": " / ".join(rec.get("facts") or []), "reference_answer": rec["reference_answer"],
                    "model_answer": g["answer"], "category": cat, "category_name": CAT[cat], "missing_facts": " / ".join(s["missing_facts"]),
                    "correct_evidence": " / ".join(f"{d['fact']}→{d['cid']}" for d in s["fact_detail"]), "basis": basis,
                    "review_status": "provisional（AI 归类，非人工）"})
    out = V5 / "results" / "error_ledger"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "error_ledger.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hn = Counter(r["subtype"] for r in read_jsonl(EXP / "p6_v4/data/v4/structured_only/train.jsonl") if r.get("hard_negative_answer"))
    summary = {"n_rows": len(rows), "categories": CAT, "by_group": {g: dict(sorted(c.items())) for g, c in cnt.items()},
               "total": dict(sorted(sum(cnt.values(), Counter()).items())),
               "v4_hard_negative_answers_by_subtype": dict(hn),
               "note": "只用于诊断错误模式；旧测试题和答案不进入任何训练或开发数据。全部 provisional。"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_rows", "by_group", "total", "v4_hard_negative_answers_by_subtype")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
