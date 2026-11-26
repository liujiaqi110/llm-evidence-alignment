"""超过 max_length=1024 的偏好对截断审计（CPU，只读；按 TRL 0.23.1 的实际分词与截断规则复现）。

    python -m rag_finetuning.experiments.p6_v5.diagnostics.truncation_audit

TRL 规则（dpo_trainer.py）：
* tokenize_row：prompt 只保留最后 max_prompt_length=896 个 token（左截断）；回答末尾加 eos，不截断（max_completion_length=None）；
* concatenated_forward，truncation_mode 默认 keep_end：chosen / rejected 各自的 [prompt + 回答] 长度 L 超过 1024 时，
  去掉序列**开头**的 L − 1024 个 token。所以 chosen 和 rejected 被截掉的 prompt 长度可能不同。
输出：diagnostics/truncation/truncation_audit.json、truncated_pairs.csv、worst_case_sets.json（供最坏情况显存测试使用）。
"""
from __future__ import annotations

import csv
import json
import statistics

from rag_finetuning.evaluation.scoring import normalize
from rag_finetuning.experiments.p6_v5.diagnostics.dpo_diag_common import MAX_LEN, MAX_PROMPT, V5, tokenized_pairs, tokenizer

OUT = V5 / "diagnostics" / "truncation"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    rows, tk = tokenized_pairs(tok)
    _, tk_full = tokenized_pairs(tok, max_prompt_length=None)
    info = []
    for r, t, tf in zip(rows, tk, tk_full):
        p, c, j = len(t["prompt_input_ids"]), len(t["chosen_input_ids"]), len(t["rejected_input_ids"])
        cut_c, cut_r = max(0, p + c - MAX_LEN), max(0, p + j - MAX_LEN)
        info.append({"pair_id": r["pair_id"], "type": r["type"], "rejected_kind": r["rejected_kind"], "origin": r["origin"],
                     "prompt": p, "prompt_untruncated": len(tf["prompt_input_ids"]), "chosen": c, "rejected": j,
                     "seq_chosen": p + c, "seq_rejected": p + j, "cut_chosen": cut_c, "cut_rejected": cut_r, "row": r, "ids": t["prompt_input_ids"]})
    prompt_trunc = [x for x in info if x["prompt_untruncated"] > MAX_PROMPT]
    trunc = [x for x in info if x["cut_chosen"] or x["cut_rejected"]]
    recs = []
    for x in trunc:
        r, ids = x["row"], x["ids"]
        kept_c = tok.decode(ids[x["cut_chosen"]:])
        kept_r = tok.decode(ids[x["cut_rejected"]:])
        cut_text = tok.decode(ids[:max(x["cut_chosen"], x["cut_rejected"])])

        def lost(kept):
            n = normalize(kept)
            ev_lost = [e["cid"] for e in r["evidence"] if normalize(e["text"].split("\n", 1)[-1][:40]) not in n]
            fact_cids = set(c for c in (r.get("fact_cids") or []) if c)
            facts_lost = [f for f in r["required_facts"] if not any(a and normalize(a) in n for a in f.split("|"))]
            return ev_lost, sorted(set(ev_lost) & fact_cids), facts_lost, normalize(r["question"]) in n
        evc, gfc, flc, qc = lost(kept_c)
        evr, gfr, flr, qr = lost(kept_r)
        recs.append({"pair_id": x["pair_id"], "type": x["type"], "rejected_kind": x["rejected_kind"], "origin": x["origin"],
                     "prompt_tokens": x["prompt"], "chosen_tokens": x["chosen"], "rejected_tokens": x["rejected"],
                     "seq_chosen": x["seq_chosen"], "seq_rejected": x["seq_rejected"], "cut_chosen": x["cut_chosen"], "cut_rejected": x["cut_rejected"],
                     "cut_difference": abs(x["cut_chosen"] - x["cut_rejected"]),
                     "system_prompt_cut": "<|im_start|>system" in cut_text,
                     "cut_text_head": cut_text[:200].replace("\n", "⏎"),
                     "evidence_lost_chosen": evc, "evidence_lost_rejected": evr,
                     "fact_source_lost_chosen": gfc, "fact_source_lost_rejected": gfr,
                     "facts_absent_from_kept_prompt_chosen": flc, "facts_absent_from_kept_prompt_rejected": flr,
                     "question_kept": qc and qr, "same_prompt_seen_by_both": x["cut_chosen"] == x["cut_rejected"]})
    with open(OUT / "truncated_pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        for z in recs:
            w.writerow({k: (" / ".join(v) if isinstance(v, list) else v) for k, v in z.items()})
    by_seq = sorted(info, key=lambda x: -max(x["seq_chosen"], x["seq_rejected"]))
    by_comp = sorted(info, key=lambda x: -max(x["chosen"], x["rejected"]))
    keep = [x for x in by_seq if max(x["seq_chosen"], x["seq_rejected"]) <= MAX_LEN]
    keep_sum = sorted(keep, key=lambda x: -(x["seq_chosen"] + x["seq_rejected"]))
    sets = {"longest_seq_8": [x["pair_id"] for x in by_seq[:8]], "longest_completion_8": [x["pair_id"] for x in by_comp[:8]],
            "longest_untruncated_8": [x["pair_id"] for x in keep_sum[:8]],
            "longest_completion_untruncated_8": [x["pair_id"] for x in sorted(keep, key=lambda x: -(x["chosen"] + x["rejected"]))[:8]]}
    sets["detail"] = {k: [{f: x[f] for f in ("pair_id", "prompt", "chosen", "rejected", "seq_chosen", "seq_rejected")} for x in v[:8]]
                      for k, v in (("longest_seq_8", by_seq), ("longest_completion_8", by_comp), ("longest_untruncated_8", keep_sum), ("longest_completion_untruncated_8", sorted(keep, key=lambda x: -(x["chosen"] + x["rejected"]))))}
    (OUT / "worst_case_sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=2), encoding="utf-8")
    tot = [max(x["seq_chosen"], x["seq_rejected"]) for x in info]
    rep = {"n_pairs": len(info), "max_length": MAX_LEN, "max_prompt_length": MAX_PROMPT, "truncation_mode": "keep_end（TRL 默认）",
           "prompt_over_896": len(prompt_trunc),
           "seq_len_stats": {"median": statistics.median(tot), "p95": sorted(tot)[int(.95 * len(tot))], "max": max(tot)},
           "pairs_truncated": len(recs), "sequences_truncated": sum((z["cut_chosen"] > 0) + (z["cut_rejected"] > 0) for z in recs),
           "system_prompt_cut": sum(z["system_prompt_cut"] for z in recs),
           "pairs_with_different_prompt_seen": sum(not z["same_prompt_seen_by_both"] for z in recs),
           "max_cut_tokens": max(max(z["cut_chosen"], z["cut_rejected"]) for z in recs),
           "pairs_losing_any_evidence_chunk": sum(bool(z["evidence_lost_chosen"] or z["evidence_lost_rejected"]) for z in recs),
           "pairs_losing_fact_source_chunk": sum(bool(z["fact_source_lost_chosen"] or z["fact_source_lost_rejected"]) for z in recs),
           "pairs_question_lost": sum(not z["question_kept"] for z in recs),
           "by_type": {t: sum(z["type"] == t for z in recs) for t in sorted({z["type"] for z in recs})},
           "by_kind": {k: sum(z["rejected_kind"] == k for z in recs) for k in sorted({z["rejected_kind"] for z in recs})},
           "worst_case_sets": {k: v for k, v in sets.items() if k != "detail"}}
    (OUT / "truncation_audit.json").write_text(json.dumps({"summary": rep, "pairs": recs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
