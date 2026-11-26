"""辅助 LLM-as-Judge（qwen2.5:7b，provisional）：对 test_v3 的应回答 / 部分回答 / 冲突题判断“忠实”与“完整”。

    python -m rag_finetuning.experiments.p6_v3.scripts.judge_v3 --groups A,B,C,D,E --out <dir>/judge.json

* 评审模型 = F 组模型本身，因此 F 组不参与（不让模型评自己的答案）。
* 不告诉评审是哪个组的答案；复用 P6-v1 的评审提示（evaluation/llm_judge.py）。
* 结果只作辅助证据，与规则评分、人工审核分开报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.evaluation.llm_judge import judge_one
from rag_finetuning.compat.local_llm import OllamaClient

V3 = Path(__file__).resolve().parents[1]
EVAL = Path("workspace/runs/p6_v3/eval/test_v3")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="A,B,C,D,E")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = [r for r in read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl") if r["behavior"] != "refuse"]
    c = OllamaClient(model="qwen2.5:7b", timeout_s=600)
    info = c.model_info()
    res = {"label": "llm_judge（qwen2.5:7b，provisional，非人工）", "judge": {k: info.get(k) for k in ("model", "digest", "quantization")},
           "items": {}, "summary": {}}
    for g in a.groups.split(","):
        if g == "F":
            raise SystemExit("F 组即评审模型本身，不参与评审")
        gens = {x["id"]: x for x in read_jsonl(EVAL / g / "generations.jsonl")}
        out = {r["id"]: judge_one(c, r, gens[r["id"]]["answer"]) for r in recs}
        res["items"][g] = out
        v = list(out.values())
        res["summary"][g] = {k: {lvl: sum(1 for x in v if x[k] == lvl) for lvl in ("yes", "partly", "no", None)} for k in ("faithful", "complete")}
        res["summary"][g]["n"] = len(v)
        print(g, json.dumps(res["summary"][g], ensure_ascii=False), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
