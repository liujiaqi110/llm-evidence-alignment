"""Collect the versionable experiment artefacts from the run directory (outside Git) into rag_finetuning/reports/results/.

    python -m rag_finetuning.scripts.collect_reports --runs workspace/runs

Copies only aggregates, per-question score tables and run records (no weights, checkpoints or optimizer state);
also derives the training loss curves, judge-vs-rule agreement and the base→candidate case lists used by REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rag_finetuning.common import REPO, ROOT, read_jsonl

OUT = ROOT / "reports" / "results"
RUNS = {"v1": "qlora_r8_v1", "v2": "qlora_r8_v2_noderived", "smoke10": "qlora_r8_v1_smoke10"}


def evaluate(runs: Path, pair: str, out: Path, split="test"):
    names = {"base": f"{split}_base_nf4", "v1": f"{split}_qlora_r8_v1", "v2": f"{split}_qlora_r8_v2"}
    if split == "test":
        names["ref7b"] = "test_ollama_7b"
    args = [sys.executable, "-m", "rag_finetuning.evaluation.evaluate", "--split", split, "--pair", pair,
            "--review-sheet", str(ROOT / "reports" / "test_review_sheet.csv"), "--out", str(out)]
    for n, d in names.items():
        args += ["--run", f"{n}={runs / 'eval' / d}"]
    subprocess.run(args, check=True, cwd=REPO, capture_output=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = runs / "eval" / "_collect"
    for pair in ("base,v1", "base,v2", "v1,v2"):
        evaluate(runs, pair, tmp / pair.replace(",", "_vs_"))
    evaluate(runs, "base,v1", tmp / "val", split="val")
    shutil.copy(tmp / "base_vs_v1" / "summary.json", OUT / "test_summary_base_vs_v1.json")
    shutil.copy(tmp / "base_vs_v2" / "summary.json", OUT / "test_summary_base_vs_v2.json")
    shutil.copy(tmp / "v1_vs_v2" / "summary.json", OUT / "test_summary_v1_vs_v2.json")
    shutil.copy(tmp / "base_vs_v1" / "per_question.csv", OUT / "test_per_question.csv")
    shutil.copy(tmp / "val" / "summary.json", OUT / "val_summary.json")
    for pair in ("base_vs_v1", "base_vs_v2"):
        ch = json.loads((tmp / pair / "changes.json").read_text(encoding="utf-8"))
        slim = {k: [{kk: vv for kk, vv in it.items()} for it in ch[k]] for k in ("improved", "regressed")}
        slim["unchanged_ids"] = [it["id"] for it in ch["unchanged"]]
        slim["error_counts"] = ch["error_counts"]
        (OUT / f"changes_{pair}.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    # training records + loss curves
    rec_out = {}
    for k, r in RUNS.items():
        d = runs / r
        if not (d / "run_record.json").exists():
            continue
        rec = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
        log = read_jsonl(d / "train_log.jsonl")
        rec["loss_curve"] = [{"step": x["step"], "epoch": round(x["epoch"], 3), "loss": x["loss"],
                              "grad_norm": round(x["grad_norm"], 4), "lr": x["learning_rate"]} for x in log if "loss" in x]
        rec["eval_curve"] = [{"epoch": round(x["epoch"] or 0.0, 3), "eval_loss": x["eval_loss"]} for x in log if "eval_loss" in x]
        acl = d / "acl_regression.json"
        if acl.exists():
            rec["acl_regression"] = [{kk: v[kk] for kk in ("model", "answers", "generations", "statuses", "ok", "violations")}
                                     for v in json.loads(acl.read_text(encoding="utf-8"))]
        rec_out[k] = rec
    (OUT / "training_runs.json").write_text(json.dumps(rec_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # judge (provisional) + agreement with rule-based completeness
    from rag_finetuning.evaluation.scoring import score_item
    test = {r["id"]: r for r in read_jsonl(ROOT / "data" / "built" / "test.jsonl")}
    j = json.loads((runs / "eval" / "judge_test.json").read_text(encoding="utf-8"))
    agree = {}
    for name, items in j["items"].items():
        gdir = {"base": "test_base_nf4", "v1": "test_qlora_r8_v1", "v2": "test_qlora_r8_v2"}[name]
        gens = {g["id"]: g for g in read_jsonl(runs / "eval" / gdir / "generations.jsonl")}
        same = n = 0
        for qid, v in items.items():
            if v["complete"] is None:
                continue
            rule = score_item(test[qid], gens[qid]["answer"])["complete"]
            n += 1
            same += int(bool(rule) == (v["complete"] == "yes"))
        agree[name] = {"n": n, "agree": same, "rate": round(same / n, 3) if n else None}
    (OUT / "llm_judge_test.json").write_text(json.dumps({"label": j["label"], "judge": {k: j["judge"].get(k) for k in ("model", "digest", "quantization")},
                                                         "summary": j["summary"], "rule_vs_judge_completeness_agreement": agree,
                                                         "items": j["items"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    e2e = {}
    for n in ("base", "v1", "v2", "ref7b"):
        p = runs / "e2e" / n / "e2e_example_qa.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            e2e[n] = {"summary": d["summary"], "rows": [{k: r[k] for k in r if k != "answer"} | {"answer": r["answer"]} for r in d["rows"]]}
    (OUT / "e2e_example_qa.json").write_text(json.dumps(e2e, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(ROOT / "data" / "built" / "manifest.json", OUT / "dataset_manifest.json")
    print("collected into", OUT)
    print(json.dumps(agree))
    return 0


if __name__ == "__main__":
    sys.exit(main())
