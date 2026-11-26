"""Auxiliary LLM-as-judge for faithfulness / completeness (provisional; never a substitute for human review).

    python -m rag_finetuning.evaluation.llm_judge --split test --judge qwen2.5:7b \
        --run base=<dir> --run qlora=<dir> --out <dir>/judge.json

* Judge = local Ollama model; it is NEVER used on its own answers (pass only runs from other models).
* Only items whose expected behaviour is answer / partial / conflict are judged (refusals are rule-scored).
* The judge sees question, evidence, reference answer and the candidate answer with citation IDs stripped of
  brackets, and returns JSON: {"faithful": "yes|partly|no", "complete": "yes|partly|no", "unsupported": [...]}.
* Answers are judged in a model-blind way (the judge is not told which model wrote the answer).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_finetuning.common import BUILT, read_jsonl
from rag_finetuning.compat.local_llm import OllamaClient

JUDGE_SYSTEM = """你是一名严格的评审员，负责检查知识库问答系统的回答。只依据给出的【资料】判断，不使用任何外部知识。
请输出 JSON：{"faithful": "yes|partly|no", "complete": "yes|partly|no", "unsupported": ["……"]}
- faithful：回答中的每一个事实陈述是否都能在【资料】中找到依据。全部有依据为 yes；有少量无依据或与资料不符的内容为 partly；主要结论无依据或与资料矛盾为 no。如果回答是拒答（说资料无法回答），但资料其实包含答案，faithful 记为 yes，complete 记为 no。
- complete：与【参考答案】相比，回答是否覆盖了回答问题所必需的全部要点（措辞可以不同）。
- unsupported：列出回答中没有资料依据或与资料矛盾的陈述（原文摘录，最多 3 条），没有则为空列表。
只输出 JSON，不要输出其他内容。"""


def judge_one(client: OllamaClient, rec: dict, answer: str) -> dict:
    ev = "\n\n".join(f"[{e['cid']}] {e['text']}" for e in rec["evidence"])
    user = (f"【资料】\n{ev}\n\n【问题】\n{rec['question']}\n\n【参考答案】\n{rec['reference_answer']}\n\n"
            f"【待评回答】\n{answer}")
    g = client.chat([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                    {"temperature": 0, "seed": 42, "num_predict": 256, "num_ctx": 4096}, fmt="json")
    try:
        v = json.loads(g.text)
    except json.JSONDecodeError:
        return {"faithful": None, "complete": None, "unsupported": [], "judge_error": "bad_json"}
    norm = lambda x: x if x in ("yes", "partly", "no") else None  # noqa: E731
    return {"faithful": norm(v.get("faithful")), "complete": norm(v.get("complete")),
            "unsupported": [str(u) for u in (v.get("unsupported") or [])][:3]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--judge", default="qwen2.5:7b")
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    recs = [r for r in read_jsonl(BUILT / f"{a.split}.jsonl") if r["behavior"] != "refuse"]
    client = OllamaClient(model=a.judge, timeout_s=600)
    info = client.model_info()
    res = {"judge": info, "label": "llm_judge (provisional, not human-reviewed)", "items": {}, "summary": {}}
    for spec in a.run:
        name, d = spec.split("=", 1)
        meta = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
        if a.judge in str(meta.get("model")):
            raise SystemExit(f"refusing to let {a.judge} judge its own answers ({name})")
        gens = {g["id"]: g for g in read_jsonl(Path(d) / "generations.jsonl")}
        out = {}
        for r in recs:
            out[r["id"]] = judge_one(client, r, gens[r["id"]]["answer"])
        res["items"][name] = out
        vals = list(out.values())
        res["summary"][name] = {k: {lvl: sum(1 for v in vals if v[k] == lvl) for lvl in ("yes", "partly", "no", None)}
                                for k in ("faithful", "complete")}
        res["summary"][name]["n"] = len(vals)
        print(name, json.dumps(res["summary"][name], ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
