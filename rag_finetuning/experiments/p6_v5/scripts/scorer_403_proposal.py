"""评分器修订 4.0.3 的影响评估模块（规则已于 2026-11-24 并入正式评分器 scoring_v4.py；本模块以 4.0.2 为基线对照）。

提议内容只涉及引用 ID 的识别，不改变内容判断（content_ok_v4）、必要事实匹配和任何门槛：
  P1 非十六进制畸形 ID：在引用区域（方括号内、“引用：”行）中，长度 6～20、含 ≥ 2 个数字、十六进制字符占比 ≥ 0.6 的字母数字串，
     如果不是合法的 12 位小写十六进制 ID、也不是内部事实编号（F1 等），记为畸形（例：d3a46887aml、72e16b3La4d0f5、d979ml）；
  P2 大写 12 位十六进制（例：2A249B096905）：现在被两条正则都忽略，提议记为畸形（不做大小写归一，因为提示词要求原样使用编号）；
  P3 12 位十六进制夹在更长的字母数字串中（例：15c4744La1b2c3d4e5f6）：现在 CITE_RE 会从中截出 12 位当作引用，提议整个串记为畸形，
     不再截出引用。
效果：这些情况 citation_valid = False、format_ok = False；content_ok_v4 不变（4.0.1 起引用问题不影响内容判断）。

    python -m rag_finetuning.experiments.p6_v5.scripts.scorer_403_proposal     # 影响评估（只读）
"""
from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4

V5 = Path(__file__).resolve().parents[1]
OUT = V5 / "diagnostics" / "scorer_review"
# 2026-11-24：规则已作为修订 4 并入正式评分器 scoring_v4.py（4.0.3）；此处重新导出，保留本模块用于影响评估复现
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import citation_regions, classify_token, malformed_403  # noqa: E402,F401


def score_item_403_proposed(rec: dict, answer: str) -> dict:
    """4.0.2 结果 + 提议规则的引用判定（只用于影响评估）。"""
    s = dict(score_item_v4(rec, answer, scorer="4.0.2"))
    extra = [t for t, k in malformed_403(answer) if not k.startswith("hex_wrong_length")]
    s["malformed_403_extra"] = extra
    s["citation_valid_403"] = s["citation_valid"] and not extra
    s["format_ok_403"] = s["format_ok"] and not extra
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summ = [], {}
    hd = {r["id"]: r for r in read_jsonl(V5 / "data/hard_dev/built/hard_dev.jsonl")}
    for p in sorted(glob.glob("workspace/runs/p6_v5/eval/hard_dev/*/generations.jsonl")):
        g = Path(p).parent.name
        c = Counter()
        for x in read_jsonl(p):
            s = score_item_403_proposed(hd[x["id"]], x["answer"])
            c["n"] += 1
            c["citation_valid_402"] += s["citation_valid"]
            c["citation_valid_403"] += s["citation_valid_403"]
            c["format_ok_402"] += s["format_ok"]
            c["format_ok_403"] += s["format_ok_403"]
            c["content_ok_v4（不受影响）"] += s["content_ok_v4"]
            if s["malformed_403_extra"]:
                rows.append({"source": f"hard_dev:{g}", "id": x["id"], "tokens": " ".join(s["malformed_403_extra"]),
                             "kinds": " ".join(k for _, k in malformed_403(x["answer"])), "answer_excerpt": x["answer"][:300]})
        summ[f"hard_dev:{g}"] = dict(c)
    pairs = read_jsonl(V5 / "data/train_pref/dpo_pairs.jsonl")
    pc = Counter()
    for p in pairs:
        for side in ("chosen", "rejected"):
            m = [t for t, k in malformed_403(p[side]) if not k.startswith("hex_wrong_length")]
            if m:
                pc[f"{side}:{p['rejected_kind'] if side == 'rejected' else p['chosen_source']}"] += 1
                rows.append({"source": f"dpo_pairs:{side}", "id": p["pair_id"], "tokens": " ".join(m),
                             "kinds": " ".join(k for _, k in malformed_403(p[side])), "answer_excerpt": p[side][:300]})
    summ["dpo_pairs_with_extra_malformed"] = dict(pc)
    cand = Counter()
    for g in ("A", "B", "C42"):
        for x in read_jsonl(V5 / f"data/train_pref/candidates_{g}.jsonl"):
            for _, k in malformed_403(x["answer"]):
                cand[f"{g}:{k}"] += 1
    summ["training_candidates_token_counts"] = dict(cand)
    with open(OUT / "malformed_403_cases.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "id", "tokens", "kinds", "answer_excerpt"])
        w.writeheader()
        w.writerows(rows)
    (OUT / "malformed_403_impact.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
