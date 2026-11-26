"""P6-v5 偏好对构造与审计（读取 build_pref_ef 的产物；只使用训练侧数据）。

    python -m rag_finetuning.experiments.p6_v5.scripts.build_pref_pairs

规则：
* chosen：优先用 3B 模型自己写出的完整正确答案（v3 > v4 > Base），否则用构造保证正确的参考答案；必须通过评分器 v4 的完整检查；
* rejected：优先用 3B Base / v3 / v4 的真实错误输出，且必须“有明确、可解释的错误”且不是胡言乱语（无无效 ID、无证据外数字）；
  没有真实错误时才用人工构造（只删减句子或替换引用，不编造事实），两类分别标记；
* 排除：与 chosen 只有引用 / 格式差异；语义等价或偏好不明确（rejected 在内容、事实覆盖、引用、陈述支持上都不比 chosen 差）；
* 每个问题最多 2 对；
* 长度偏差控制：chosen 比 rejected 长的对占比不超过 MAX_CHOSEN_LONGER（预先设定 0.65），超出时优先删除长度差最大的人工构造对。
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rag_finetuning.common import read_jsonl, write_jsonl
from rag_finetuning.evaluation.scoring import normalize
from rag_finetuning.experiments.p6_v5.scripts.build_pref_v5 import construct
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
EXP = V5.parent
OUT = V5 / "data" / "train_pref"
TYPE_NAME = {"A": "完整 vs 遗漏必要条件", "B": "完整 vs 只用单一证据", "C": "正确例外处理 vs 忽略例外", "D": "正确版本判断 vs 混用版本",
             "E": "有依据的部分回答 vs 过度拒答", "F": "正确拒答 vs 无依据推断", "G": "识别真冲突 vs 只采信一方",
             "H": "正确引用 vs 合法但错误的引用", "I": "范围 / 条件不同 vs 误判为冲突"}
MAX_CHOSEN_LONGER = 0.65


def body(a):
    return re.split(r"\n?\s*引用\s*[：:]", a or "", maxsplit=1)[0]


def body_len(a):
    return len(re.sub(r"\s", "", body(a)))


def plain(a):
    return normalize(re.sub(r"\[[0-9a-f]{12}\]", "", body(a)))


def classify_real(rec, s):
    if rec["behavior"] == "refuse" and not s["refusal_ok"]:
        return "F"
    if rec["behavior"] in ("answer", "partial", "conflict") and s["whole_refusal_on_answerable"]:
        return "E"
    if rec["behavior"] == "conflict":
        return "G"
    if rec.get("conflict_kind") in ("version_override", "old_rule_still_applies") and not s["content_ok_v4"]:
        return "D"
    if rec.get("no_conflict") and s.get("conflict_claimed"):
        return "I"
    if rec.get("subtype") == "exception" and not s["content_ok_v4"]:
        return "C"
    if s["missing_facts"]:
        return "B" if (s["first_evidence_only"] or s["last_evidence_only"]) else "A"
    if s["content_ok_v4"] and (s["fact_attribution_ok"] is False or s["claim_support_full"] is False):
        return "H"
    return None


def full_ok(s):
    return s["content_ok_v4"] and s["format_ok"] and s["claim_support_full"] in (True, None) and s["fact_attribution_ok"] is not False \
        and s["multi_complete_v5"] in (True, None)


def main() -> int:
    nat, st = read_jsonl(OUT / "natural_ef.jsonl"), read_jsonl(OUT / "structured.jsonl")
    cands = defaultdict(list)
    for g in ("A", "B", "C42"):
        p = OUT / f"candidates_{g}.jsonl"
        if p.exists():
            for x in read_jsonl(p):
                cands[x["id"]].append(x)
    pairs, excluded, real_stats = [], [], Counter()
    for rec in nat + st:
        if not score_item_v4(rec, rec["reference_answer"])["content_ok_v4"]:
            excluded.append({"question_id": rec["id"], "source": "reference", "type": "", "reason": "参考答案自检未通过，整题不用"})
            continue
        chosen_pool, real = [], []
        for x in cands.get(rec["id"], []):
            s = score_item_v4(rec, x["answer"])
            real_stats[f"{x['group']}:{'正确完整' if full_ok(s) else '有错误'}"] += 1
            if full_ok(s):
                chosen_pool.append(x)
                continue
            if not s["citation_valid"] or s["unsupported_numbers_v4"] or len(plain(x["answer"])) < 4:
                excluded.append({"question_id": rec["id"], "source": f"real:{x['group']}", "type": "",
                                 "reason": "真实输出含无效 ID / 证据外数字 / 过短，不作 rejected"})
                continue
            t = classify_real(rec, s)
            if t is None:
                excluded.append({"question_id": rec["id"], "source": f"real:{x['group']}", "type": "", "reason": "真实输出错误类型不明确，不使用"})
                continue
            miss = " / ".join(s["missing_facts"])
            real.append((t, x["answer"], f"模型真实输出（{x['group']}）：" + (f"缺失 {miss}" if miss else "行为或引用错误"), f"real:{x['group']}"))
        if chosen_pool:
            pick = sorted(chosen_pool, key=lambda x: ["B", "C42", "A"].index(x["group"]))[0]
            chosen, chosen_src = pick["answer"], f"model:{pick['group']}"
        else:
            chosen, chosen_src = rec["reference_answer"], "reference"
        cons = [(t, r, h, "constructed") for t, r, h in construct(rec)]
        has_real_core = any(t in "ABCDEFGI" for t, *_ in real)
        pool = real + ([c for c in cons if c[0] != "H"] if not has_real_core else []) + [c for c in cons if c[0] == "H"][:1]
        sch = score_item_v4(rec, chosen)
        n_added = 0
        for t, r, h, src in sorted(pool, key=lambda z: (z[3] == "constructed", z[0] == "H")):
            if n_added >= 2:
                break
            srj = score_item_v4(rec, r)
            if plain(r) == plain(chosen):
                excluded.append({"question_id": rec["id"], "source": src, "type": t, "reason": "与 chosen 只有引用 / 格式差异"})
                continue
            same = (srj["content_ok_v4"] == sch["content_ok_v4"] and (srj["required_fact_coverage"] or 0) >= (sch["required_fact_coverage"] or 0)
                    and srj["fact_attribution_ok"] is not False and srj["claim_support_full"] is not False
                    and (rec["behavior"] != "refuse" or srj["refusal_ok"]))
            if same:
                excluded.append({"question_id": rec["id"], "source": src, "type": t, "reason": "rejected 与 chosen 语义等价或偏好不明确"})
                continue
            if not sch["content_ok_v4"]:
                excluded.append({"question_id": rec["id"], "source": src, "type": t, "reason": "chosen 未通过检查"})
                continue
            n_added += 1
            pairs.append({"pair_id": f"{rec['id']}__{len(pairs)}", "question_id": rec["id"], "type": t, "type_name": TYPE_NAME[t],
                          "origin": rec.get("origin") or "natural_evidence_first", "subtype": rec.get("subtype"), "question": rec["question"],
                          "evidence": rec["evidence"], "chosen": chosen, "chosen_source": chosen_src, "rejected": r, "rejected_source": src,
                          "rejected_kind": "真实模型错误" if src.startswith("real") else "人工构造（只删减 / 替换，不编造）",
                          "error_explanation": h, "required_facts": rec["facts"], "fact_cids": rec.get("fact_cids"),
                          "source_docs": rec.get("source_docs"), "construction": rec.get("generation_method"),
                          "chosen_fact_coverage": sch["required_fact_coverage"], "rejected_fact_coverage": srj["required_fact_coverage"],
                          "rejected_missing_facts": srj["missing_facts"], "chosen_len": body_len(chosen), "rejected_len": body_len(r),
                          "label_status": "auto_checked_provisional（未经人工复核）"})
    longer = lambda p: p["chosen_len"] > p["rejected_len"]
    n_before = len(pairs)
    while pairs and sum(map(longer, pairs)) / len(pairs) > MAX_CHOSEN_LONGER:
        cand = sorted([p for p in pairs if longer(p)], key=lambda p: (p["rejected_kind"] == "真实模型错误", -(p["chosen_len"] - p["rejected_len"])))
        x = cand[0]
        pairs.remove(x)
        excluded.append({"question_id": x["question_id"], "source": x["rejected_source"], "type": x["type"], "reason": "长度偏差控制删除"})
    write_jsonl(OUT / "dpo_pairs.jsonl", pairs)
    write_jsonl(OUT / "excluded.jsonl", excluded)
    v3 = read_jsonl(EXP / "p6_v3/data/v3/full/train.jsonl")
    write_jsonl(OUT / "sft_train.jsonl", v3 + nat)
    # 审计表（人工字段留空）
    with open(V5 / "PREFERENCE_PAIR_AUDIT.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["pair_id", "question_id", "type", "type_name", "origin", "construction", "source_docs", "question", "required_facts", "fact_cids",
                "chosen_source", "chosen", "rejected_kind", "rejected_source", "rejected", "error_explanation", "rejected_missing_facts",
                "chosen_fact_coverage", "rejected_fact_coverage", "chosen_len", "rejected_len", "label_status",
                "human_chosen_correct", "human_rejected_has_clear_error", "human_preference_clear", "human_note", "human_approved"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in pairs:
            row = {k: p.get(k) for k in cols if k in p}
            row.update({"required_facts": " / ".join(p["required_facts"]), "fact_cids": " / ".join(c or "" for c in (p["fact_cids"] or [])),
                        "rejected_missing_facts": " / ".join(p["rejected_missing_facts"]), "source_docs": ",".join(p.get("source_docs") or []),
                        "human_chosen_correct": "", "human_rejected_has_clear_error": "", "human_preference_clear": "", "human_note": "", "human_approved": ""})
            w.writerow(row)
    rep = {"pairs": len(pairs), "pairs_before_length_control": n_before, "questions_with_pairs": len({p["question_id"] for p in pairs}),
           "questions_total": len(nat) + len(st), "natural_questions": len(nat), "structured_questions": len(st),
           "by_type": {f"{k} {TYPE_NAME[k]}": v for k, v in sorted(Counter(p["type"] for p in pairs).items())},
           "by_rejected_kind": dict(Counter(p["rejected_kind"] for p in pairs)), "by_rejected_source": dict(Counter(p["rejected_source"] for p in pairs)),
           "by_chosen_source": dict(Counter(p["chosen_source"] for p in pairs)), "by_origin": dict(Counter(p["origin"] for p in pairs)),
           "by_origin_and_kind": dict(Counter(f"{p['origin']} | {p['rejected_kind']}" for p in pairs)),
           "candidate_outputs": dict(real_stats),
           "length": {"chosen_mean": round(statistics.mean(p["chosen_len"] for p in pairs), 1), "rejected_mean": round(statistics.mean(p["rejected_len"] for p in pairs), 1),
                      "share_chosen_longer": round(sum(map(longer, pairs)) / len(pairs), 3), "cap": MAX_CHOSEN_LONGER,
                      "dropped_for_length_control": n_before - len(pairs)},
           "excluded_by_reason": dict(Counter(e["reason"] for e in excluded)),
           "sft_train": len(v3) + len(nat), "sft_v3_part": len(v3), "sft_new_hard_multi": len(nat),
           "sha256": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in ("dpo_pairs.jsonl", "sft_train.jsonl")}}
    (OUT / "pairs_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
