"""人工审核材料：分层抽样 + 本地 HTML 审核页面（所有人工字段留空；AI / 程序判定单独标注，不代替人工）。

    python -m rag_finetuning.experiments.p6_v5.scripts.make_review_dashboard

配额（DPO_EXPERIMENT_PLAN 第 3 节预设）：
  Q1 H 类偏好对 ≥ 60：真实错误、人工构造各 32，组内按（来源 natural/structured × “rejected 覆盖率 ≥ chosen”）分层，最大余数法按比例分配，每个非空层至少 2；
  Q2 chosen 为程序参考答案 ≥ 40：抽 44，从未被 Q1 / Q3 选中的参考答案对中按（来源 × rejected 类别）分层抽取，保证 44 道不与其他配额重复；
     已在 Q1 / Q3 中、且 chosen 也是参考答案的对，额外记为“同时满足 Q2”；
  Q3 C / I 类人工构造对：全部（59）；
  Q4 hard_dev 高优先级：HUMAN_REVIEW_CHECKLIST.csv 中 priority = 高 的全部（24）；
  Q5 评分争议补充：争议清单中出现、但不在 Q4 的 hard_dev 题（全部；配额之外单列）。
去重：同一偏好对只审一次，记录它满足的所有配额。随机种子 20261124；抽样记录写入 data/review/sampling_record.json。
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.scoring import normalize
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as v2
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4

V5 = Path(__file__).resolve().parents[1]
TP = V5 / "data" / "train_pref"
RV = V5 / "data" / "review"
SEED = 20261124
TRUNC = V5 / "diagnostics" / "truncation" / "truncation_audit.json"


def alloc(strata: dict, total: int, minimum: int = 2) -> dict:
    """最大余数法按比例分配，每个非空层至少 minimum（不超过层大小）。"""
    n = sum(len(v) for v in strata.values())
    base = {k: min(len(v), max(minimum, int(total * len(v) / n))) for k, v in strata.items() if v}
    rem = sorted(strata, key=lambda k: -(total * len(strata[k]) / n - int(total * len(strata[k]) / n)))
    i = 0
    while sum(base.values()) < total and any(base[k] < len(strata[k]) for k in base):
        k = rem[i % len(rem)]
        if base.get(k, 0) < len(strata[k]):
            base[k] += 1
        i += 1
    while sum(base.values()) > total:
        k = max(base, key=lambda z: base[z] - minimum)
        base[k] -= 1
    return base


def fact_sentence(fact, text):
    for s in re.split(r"(?<=[。；])", text.split("\n", 1)[-1]):
        if any(a and normalize(a) in normalize(s) for a in fact.split("|")):
            return s.strip()
    return ""


def auto(rec, ans):
    s = score_item_v4(rec, ans)
    claims = v2.score_item_v2(rec, ans)["claims"]
    return {"content_ok_v4": s["content_ok_v4"], "required_fact_coverage": s["required_fact_coverage"], "missing_facts": s["missing_facts"],
            "multi_complete_v5": s["multi_complete_v5"], "claim_support": f"{s['n_supported']}/{s['n_claims']}",
            "fact_attribution_ok": s["fact_attribution_ok"], "citation_valid": s["citation_valid"], "malformed_ids": s["malformed"],
            "fabricated_ids": s["fabricated"], "unsupported_numbers": s["unsupported_numbers_v4"], "format_ok": s["format_ok"],
            "whole_refusal": s["whole_refusal_on_answerable"], "conflict_claimed": s["conflict_claimed"],
            "claims": [{"text": c["text"], "cites": c["cites"], "inherited": c.get("inherited", False), "supported": c.get("supported")} for c in claims]}


def main() -> int:
    RV.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(TP / "dpo_pairs.jsonl")
    recs = {r["id"]: r for r in read_jsonl(TP / "natural_ef.jsonl") + read_jsonl(TP / "structured.jsonl")}
    trunc = {z["pair_id"]: z for z in json.loads(TRUNC.read_text(encoding="utf-8"))["pairs"]} if TRUNC.exists() else {}
    rng = random.Random(SEED)
    amb = lambda p: (p["rejected_fact_coverage"] or 0) >= (p["chosen_fact_coverage"] or 0)
    org = lambda p: "natural" if p["origin"] == "natural_evidence_first" else "structured"
    kind = lambda p: "real" if p["rejected_kind"].startswith("真实") else "constructed"
    record = {"seed": SEED, "scorer": SCORER_VERSION, "pairs_file_sha256": hashlib.sha256((TP / "dpo_pairs.jsonl").read_bytes()).hexdigest(), "quotas": {}}
    picked = defaultdict(set)
    # Q3：全部 C / I 人工构造
    q3 = [p["pair_id"] for p in pairs if p["type"] in ("C", "I") and kind(p) == "constructed"]
    for x in q3:
        picked[x].add("Q3_CI_constructed_all")
    record["quotas"]["Q3_CI_constructed_all"] = {"population": len(q3), "selected": len(q3), "rule": "全部"}
    # Q1：H 类分层
    q1_sel, q1_strata_rec = [], {}
    for kd in ("real", "constructed"):
        strata = defaultdict(list)
        for p in pairs:
            if p["type"] == "H" and kind(p) == kd:
                strata[f"{kd}|{org(p)}|{'rej_cov_ge_chosen' if amb(p) else 'rej_cov_lt_chosen'}"].append(p["pair_id"])
        a = alloc(strata, 32)
        for k, ids in sorted(strata.items()):
            ids = sorted(ids)
            rng.shuffle(ids)
            q1_sel += ids[:a.get(k, 0)]
            q1_strata_rec[k] = {"population": len(ids), "selected": a.get(k, 0)}
    for x in q1_sel:
        picked[x].add("Q1_H_stratified")
    record["quotas"]["Q1_H_stratified"] = {"population": sum(p["type"] == "H" for p in pairs), "selected": len(q1_sel), "strata": q1_strata_rec,
                                           "rule": "真实 / 构造各 32；层 = 来源 × 覆盖率关系；最大余数法，每层至少 2"}
    # Q2：参考答案 chosen（与 Q1 / Q3 不重复的 44 道）
    strata = defaultdict(list)
    for p in pairs:
        if p["chosen_source"] == "reference" and p["pair_id"] not in picked:
            strata[f"{org(p)}|{kind(p)}"].append(p["pair_id"])
    a = alloc(strata, 44)
    q2_sel, q2_rec = [], {}
    for k, ids in sorted(strata.items()):
        ids = sorted(ids)
        rng.shuffle(ids)
        q2_sel += ids[:a.get(k, 0)]
        q2_rec[k] = {"population": len(ids), "selected": a.get(k, 0)}
    for x in q2_sel:
        picked[x].add("Q2_reference_chosen")
    also = [p["pair_id"] for p in pairs if p["pair_id"] in picked and p["chosen_source"] == "reference" and "Q2_reference_chosen" not in picked[p["pair_id"]]]
    for x in also:
        picked[x].add("Q2_reference_chosen(同时满足)")
    record["quotas"]["Q2_reference_chosen"] = {"population": sum(p["chosen_source"] == "reference" for p in pairs), "selected_distinct": len(q2_sel),
                                               "also_satisfied_by_Q1_Q3": len(also), "strata": q2_rec,
                                               "rule": "从未被 Q1 / Q3 选中的参考答案对中按来源 × rejected 类别分层抽 44"}
    # 偏好对审核条目
    items = []
    for p in pairs:
        if p["pair_id"] not in picked:
            continue
        r = recs[p["question_id"]]
        by = {e["cid"]: e for e in r["evidence"]}
        t = trunc.get(p["pair_id"])
        items.append({"kind": "pair", "id": p["pair_id"], "quotas": sorted(picked[p["pair_id"]]), "type": p["type"], "type_name": p["type_name"],
                      "origin": p["origin"], "subtype": p.get("subtype"), "behavior": r.get("behavior"), "question": p["question"],
                      "evidence": [{"cid": e["cid"], "source": e.get("source", ""), "text": e["text"], "fact_source": e["cid"] in set(p.get("fact_cids") or [])}
                                   for e in r["evidence"]],
                      "facts": [{"fact": f, "cid": c, "sentence": fact_sentence(f, by.get(c, {}).get("text", ""))}
                                for f, c in zip(p["required_facts"], p.get("fact_cids") or [None] * len(p["required_facts"]))],
                      "chosen": p["chosen"], "chosen_source": p["chosen_source"], "rejected": p["rejected"], "rejected_source": p["rejected_source"],
                      "rejected_kind": p["rejected_kind"], "error_explanation": p["error_explanation"],
                      "auto_chosen": auto(r, p["chosen"]), "auto_rejected": auto(r, p["rejected"]),
                      "truncation": ({"cut_chosen": t["cut_chosen"], "cut_rejected": t["cut_rejected"]} if t else None)})
    order = {"Q3_CI_constructed_all": 2, "Q1_H_stratified": 0, "Q2_reference_chosen": 1}
    items.sort(key=lambda x: (min(order.get(q, 9) for q in x["quotas"]), x["id"]))
    # hard_dev 高优先级
    with open(V5 / "HUMAN_REVIEW_CHECKLIST.csv", encoding="utf-8-sig") as f:
        allrows = list(csv.DictReader(f))
    dev = [row for row in allrows if row["priority"] == "高"]
    # Q5：评分争议补充（disputes_by_category.csv 中出现、但不在高优先级里的题，例如只在 Oracle 条件下出现争议的题）
    dsp = V5 / "diagnostics/scorer_review/disputes_by_category.csv"
    extra_ids = set()
    if dsp.exists():
        with open(dsp, encoding="utf-8-sig") as f2:
            extra_ids = {r["id"] for r in csv.DictReader(f2)} - {row["id"] for row in dev}
    q5 = [row for row in allrows if row["id"] in extra_ids]
    hd = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    for row in dev + q5:
        r = hd[row["id"]]
        gold = set(r["gold_cids"])
        by = {e["cid"]: e for e in r["evidence"]}
        ans = {k.split("_answer")[0]: v for k, v in row.items() if "_answer（" in k}
        items.append({"kind": "hard_dev", "id": row["id"], "quotas": ["Q4_hard_dev_high_priority"] if row in dev else ["Q5_scorer_dispute_supplement"], "behavior": r["behavior"], "subtype": r.get("subtype"),
                      "conflict_kind": r.get("conflict_kind"), "tags": row["tags"], "question": r["question"],
                      "evidence": [{"cid": e["cid"], "source": e.get("source", ""), "text": e["text"], "fact_source": e["cid"] in gold} for e in r["evidence"]],
                      "facts": [{"fact": f, "cid": c, "sentence": fact_sentence(f, by.get(c, {}).get("text", ""))}
                                for f, c in zip(r["facts"], r.get("fact_cids") or [None] * len(r["facts"]))],
                      "reference_answer": r["reference_answer"], "forbid": r.get("forbid") or [],
                      "answers": [{"group": g, "label": next(k for k in row if k.startswith(f"{g}_answer（"))[len(g) + 8:-1], "answer": a,
                                   "auto": auto(r, a)} for g, a in ans.items()],
                      "ai_prereview_flags": row["ai_prereview_flags（非人工）"], "groups_with_dispute": row["groups_with_dispute"]})
    record["quotas"]["Q4_hard_dev_high_priority"] = {"population": 56, "selected": len(dev), "rule": "HUMAN_REVIEW_CHECKLIST.csv priority=高 全部"}
    # AI 辅助判断（程序规则，非人工）与风险排序；只写入 item["ai"] / item["risk"]，不触碰任何人工字段
    from rag_finetuning.experiments.p6_v5.scripts.review_assist import AI_ASSIST_VERSION, dev_assist, pair_assist
    pend = {}
    pf = TP / "pending_exclusions_2026-11-24.json"
    if pf.exists():
        pj = json.loads(pf.read_text(encoding="utf-8"))
        for k, lab in (("truncated_over_1024", "超长截断（决策 3）"), ("scorer_403_malformed_rejected", "评分器修订：rejected 含畸形 ID（决策 1）")):
            for i in pj[k]["pair_ids"]:
                pend[i] = lab
    drows = defaultdict(list)
    dsp = V5 / "diagnostics/scorer_review/disputes_by_category.csv"
    if dsp.exists():
        with open(dsp, encoding="utf-8-sig") as f3:
            for r in csv.DictReader(f3):
                drows[r["id"]].append(r)
    for x in items:
        if x["kind"] == "pair":
            pair_assist(x, pend)
        else:
            dev_assist(x, hd[x["id"]], drows.get(x["id"], []), score_item_v4)
    items.sort(key=lambda x: (-x["risk"]["score"], x["kind"] != "hard_dev", x["id"]))
    for x in items:
        x["ai"]["version"] = AI_ASSIST_VERSION
    record["ai_assist"] = {"version": AI_ASSIST_VERSION, "method": "确定性程序规则（scripts/review_assist.py），非人工、非大模型；页面默认折叠，人工字段不受影响",
                           "suggestions": dict(Counter(x["ai"]["suggestion"] for x in items)),
                           "pending_exclusion_in_sample": sum(bool(x["ai"]["pending_exclusion"]) for x in items)}
    record["distinct_pair_items"] = sum(x["kind"] == "pair" for x in items)
    record["quotas"]["Q5_scorer_dispute_supplement"] = {"selected": len(q5), "ids": sorted(r["id"] for r in q5),
                                                         "rule": "diagnostics/scorer_review/disputes_by_category.csv 中出现、但不在 Q4 中的 hard_dev 题（全部）"}
    record["distinct_hard_dev_items"] = len(dev) + len(q5)
    record["selected_pair_ids"] = {x["id"]: x["quotas"] for x in items if x["kind"] == "pair"}
    (RV / "sampling_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (RV / "review_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    tpl = (V5 / "scripts" / "review_dashboard_template.html").read_text(encoding="utf-8")
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    page = tpl.replace("/*__DATA__*/null", data).replace("__META__", html.escape(
        f"种子 {SEED} · 评分器 {SCORER_VERSION} · 偏好对文件 sha256 {record['pairs_file_sha256'][:12]}… · AI 辅助 v{AI_ASSIST_VERSION} · 生成后未经人工修改"))
    (V5 / "HUMAN_REVIEW_DASHBOARD.html").write_text(page, encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "selected_pair_ids"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
