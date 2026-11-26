"""Minimal, CPU-only example: score two sets of answers with the frozen rule scorer (4.0.3) and compare them with
the source-document clustered paired bootstrap used in the final evaluation.

    python -m examples.minimal.run_example            # from the repository root

Everything here is public: hand-written fictional documents and hand-written answers (no model is called and no
experiment data is needed). The numbers it prints illustrate the pipeline only; they are not experimental results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_finetuning.compat.rag_pipeline import SYSTEM_PROMPT, build_messages
from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import paired_cluster
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4

HERE = Path(__file__).resolve().parent
METRICS = ["content_ok_v4", "multi_complete_v5", "required_fact_coverage", "citation_valid", "behavior_ok"]


def read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(items, preds):
    ans = {p["id"]: p["answer"] for p in preds}
    return [{"rec": r, "s": score_item_v4(r, ans[r["id"]])} for r in items]


def aggregate(rows):
    out = {}
    for m in METRICS:
        vals = [r["s"][m] for r in rows if r["s"].get(m) is not None]
        out[m] = {"k": round(sum(float(v) for v in vals), 4), "n": len(vals)}
    return out


def main(argv=None) -> int:
    items = read_jsonl(HERE / "items.jsonl")
    a = score(items, read_jsonl(HERE / "predictions_a.jsonl"))
    b = score(items, read_jsonl(HERE / "predictions_b.jsonl"))
    r0 = items[1]
    msgs = build_messages(r0["question"], [(e["cid"], e["text"]) for e in r0["evidence"]],
                          {e["cid"]: e["source"] for e in r0["evidence"]})
    result = {
        "scorer_version": SCORER_VERSION,
        "n_items": len(items),
        "prompt_messages": len(msgs),
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "system_A": aggregate(a),
        "system_B": aggregate(b),
        "paired_B_minus_A": {m: {k: v for k, v in paired_cluster(a, b, m, iters=2000, seed=0).items()
                                 if k in ("n", "n_clusters", "diff", "ci95_cluster_bootstrap")}
                             for m in ("content_ok_v4", "multi_complete_v5")},
        "missing_facts_A": {r["rec"]["id"]: r["s"]["missing_facts"] for r in a if r["s"]["missing_facts"]},
    }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
