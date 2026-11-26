"""生成 P6-v5 最终 DPO 偏好数据集（方案三：保守；2026-11-24 用户决定跳过剩余人工审核）。

    python -m rag_finetuning.experiments.p6_v5.scripts.build_final_pref

输入：data/train_pref/dpo_pairs.jsonl（791 对，sha256 204b40b2…，不修改）
排除（按顺序记录原因；一对可有多个原因）：
  E1 超长截断：[prompt + 回答] > 1024 token，chosen / rejected 条件化的 prompt 不一致（决策 3，TRUNCATION_AUDIT.md）
  E2 评分器 4.0.3：rejected 含畸形引用 ID，按“含无效 ID 的真实输出不作 rejected”规则排除（修订 4）
  E3 人工抽查：人工主判断为“不成立”或“不确定”的对（human_raw/human_review_pair_2026-11-24.csv）
  E4 规则 1（保守适用）：H 类真实错误对中 rejected 事实覆盖率 ≥ chosen 的全部对。
     依据是人工抽查 12 条 H 类中 7 条“不确定 / 不成立”（58% > 预设 15%）；样本不足预设的 60，属于用户决定的保守处理，不是规则的正式触发
  E5 长度偏差控制：剩余对中“chosen 更长”的占比超过预设上限 0.65 时，按 build_pref_pairs 的同一规则，
     先删除长度差最大的人工构造对
不新增、不改写任何偏好对。人工审核未完成：所有对 label_status 保持 provisional，human_approved 不填写。
输出：data/train_pref/dpo_pairs_final.jsonl、excluded_final.jsonl、final_dataset_report.json
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl, write_jsonl

V5 = Path(__file__).resolve().parents[1]
TP = V5 / "data" / "train_pref"
HUMAN = V5 / "data" / "review" / "human_raw" / "human_review_pair_2026-11-24.csv"
MAX_CHOSEN_LONGER = 0.65
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    src = TP / "dpo_pairs.jsonl"
    pairs = read_jsonl(src)
    pend = json.loads((TP / "pending_exclusions_2026-11-24.json").read_text(encoding="utf-8"))
    assert pend["source_pairs_sha256"] == sha(src)
    with open(HUMAN, encoding="utf-8-sig") as f:
        human = {r["id"]: r for r in csv.DictReader(f)}
    reasons = {p["pair_id"]: [] for p in pairs}
    for i in pend["truncated_over_1024"]["pair_ids"]:
        reasons[i].append("E1 超长截断（> 1024 token，chosen / rejected 的 prompt 不一致）")
    for i in pend["scorer_403_malformed_rejected"]["pair_ids"]:
        reasons[i].append("E2 评分器 4.0.3：rejected 含畸形引用 ID")
    for i, r in human.items():
        if r["human_preference_clear"] in ("不成立", "不确定"):
            reasons[i].append(f"E3 人工抽查：偏好{r['human_preference_clear']}")
    for p in pairs:
        if p["type"] == "H" and p["rejected_kind"].startswith("真实") and (p["rejected_fact_coverage"] or 0) >= (p["chosen_fact_coverage"] or 0):
            reasons[p["pair_id"]].append("E4 规则 1 保守适用：H 类真实对，rejected 覆盖率 ≥ chosen（偏好可能不明确）")
    keep = [p for p in pairs if not reasons[p["pair_id"]]]
    longer = lambda p: p["chosen_len"] > p["rejected_len"]
    n_before_len = len(keep)
    while keep and sum(map(longer, keep)) / len(keep) > MAX_CHOSEN_LONGER:
        cand = sorted([p for p in keep if longer(p)], key=lambda p: (p["rejected_kind"].startswith("真实"), -(p["chosen_len"] - p["rejected_len"]), p["pair_id"]))
        x = cand[0]
        keep.remove(x)
        reasons[x["pair_id"]].append("E5 长度偏差控制（chosen 更长占比 > 0.65）")
    excluded = [{"pair_id": p["pair_id"], "question_id": p["question_id"], "type": p["type"], "rejected_kind": p["rejected_kind"],
                 "rejected_source": p["rejected_source"], "reasons": reasons[p["pair_id"]]} for p in pairs if reasons[p["pair_id"]]]
    final = []
    for p in keep:
        q = dict(p)
        hr = human.get(p["pair_id"])
        q["label_status"] = "auto_checked_provisional（人工审核未完成；用户决定跳过剩余审核）"
        q["human_spot_check"] = hr["human_preference_clear"] if hr and hr["human_preference_clear"] else ""
        final.append(q)
    write_jsonl(TP / "dpo_pairs_final.jsonl", final)
    write_jsonl(TP / "excluded_final.jsonl", excluded)
    first = Counter(e["reasons"][0][:2] for e in excluded)
    allr = Counter(r[:2] for e in excluded for r in e["reasons"])
    rep = {"when": "2026-11-24", "scheme": "方案三：保守（用户选择）",
           "human_review": {"status": "未完成，用户决定跳过剩余审核", "pairs_judged": sum(bool(r["human_preference_clear"]) for r in human.values()),
                            "pairs_total_in_sample": len(human), "human_approved_filled": sum(bool(r["human_approved"]) for r in human.values()),
                            "note": "抽查判断均在看过 AI 辅助建议后作出（human_ai_viewed_before_decision = 是）；不能视为独立人工审核"},
           "source": {"file": "dpo_pairs.jsonl", "pairs": len(pairs), "sha256": sha(src)},
           "excluded": len(excluded), "excluded_by_first_reason": dict(first), "excluded_reason_mentions": dict(allr),
           "length_control_removed": n_before_len - len(keep),
           "final": {"file": "dpo_pairs_final.jsonl", "pairs": len(final), "questions": len({p["question_id"] for p in final}),
                     "sha256": sha(TP / "dpo_pairs_final.jsonl"),
                     "by_type": dict(sorted(Counter(p["type"] for p in final).items())),
                     "by_rejected_kind": dict(Counter(p["rejected_kind"] for p in final)),
                     "by_origin": dict(Counter(p["origin"] for p in final)),
                     "by_chosen_source": dict(Counter(p["chosen_source"] for p in final)),
                     "share_chosen_longer": round(sum(map(longer, final)) / len(final), 3),
                     "human_spot_checked_and_kept": sum(bool(p["human_spot_check"]) for p in final)},
           "sft_train": {"file": "sft_train.jsonl", "sha256": sha(TP / "sft_train.jsonl"), "changed": False},
           "known_limitations": ["F 类（正确拒答 vs 无依据推断）0 对：原有 2 对均被 E2 排除",
                                 "D 类只有 7 对",
                                 "H 类真实错误对被 E4 全部排除（所有 H 类真实对的 rejected 覆盖率都 ≥ chosen），最终 H 类 119 对全部为人工构造（只替换 / 删减引用）",
                                 "长度偏差控制（E5）额外删除 52 对（E 23、B 16、H 8、A 5），多于方案比较时的预估；最终 580 对",
                                 "hard_dev dv_m13、dv_m16 人工抽查标为“标注有问题”，未修改冻结数据，作为已知局限",
                                 "人工审核未完成，全部结果 provisional"]}
    (TP / "final_dataset_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
