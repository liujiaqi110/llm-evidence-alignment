"""冻结实验输入：问题 + 证据（含顺序）及其哈希。冻结文件不含参考答案，推理脚本只读取冻结文件。

    python -m rag_finetuning.experiments.p6_v2.freeze_inputs --name diagnostic_47 --src rag_finetuning/data/built/test.jsonl
    python -m rag_finetuning.experiments.p6_v2.freeze_inputs --name blind_v1 --src rag_finetuning/experiments/p6_v2/blind/built/blind.jsonl

检查（任一失败即退出）：
  * Few-shot 示例文档与被测文档无交集；
  * 任何一条 Prompt（zero / few）都不包含该题或其他被测题的参考答案；
  * 已存在的同名冻结文件内容不同则拒绝覆盖（防止悄悄改动冻结集）。
同时导出两种 Prompt 的完整示例文本（prompts/*.md），便于审阅。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v2.prompts import DEMOS, demo_docs, messages, prompt_fingerprint

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen_inputs"


def item_hash(r: dict) -> str:
    payload = json.dumps({"question": r["question"], "evidence": [[e["cid"], e["source"], e["text"]] for e in r["evidence"]]},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--src", required=True)
    a = ap.parse_args(argv)
    src = Path(a.src)
    rows = read_jsonl(src)
    docs = {e["doc"] for r in rows for e in r["evidence"]}
    overlap = docs & demo_docs()
    if overlap:
        raise SystemExit(f"Few-shot 示例文档与被测文档重叠：{sorted(overlap)}")
    refs = [r["reference_answer"] for r in rows if r["behavior"] != "refuse"]
    for r in rows:
        for v in ("zero_shot", "few_shot"):
            blob = "\n".join(m["content"] for m in messages(r, v))
            for ref in refs:
                body = ref.split("\n\n引用：")[0]
                if len(body) > 12 and body in blob:
                    raise SystemExit(f"{r['id']}/{v}: 参考答案出现在 Prompt 中")
    frozen = [{"id": r["id"], "question": r["question"],
               "evidence": [{"cid": e["cid"], "source": e["source"], "doc": e["doc"], "text": e["text"]} for e in r["evidence"]],
               "input_sha256": item_hash(r)} for r in rows]
    FROZEN.mkdir(parents=True, exist_ok=True)
    out = FROZEN / f"{a.name}.jsonl"
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in frozen)
    if out.exists() and out.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{out} 已存在且内容不同：拒绝覆盖冻结集（如确需变更，请换新名称）")
    out.write_text(text, encoding="utf-8", newline="\n")
    manifest = {"name": a.name, "source_file": str(src).replace("\\", "/"),
                "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "frozen_file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "n_items": len(frozen),
                "contains_reference_answers": False, "fewshot_docs": sorted(demo_docs()),
                "input_docs": sorted(docs), "prompts": prompt_fingerprint(),
                "items": {x["id"]: x["input_sha256"] for x in frozen}}
    (FROZEN / f"{a.name}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    pdir = HERE / "prompts"
    pdir.mkdir(exist_ok=True)
    sample = rows[0]
    for v in ("zero_shot", "few_shot"):
        msgs = messages(sample, v)
        md = [f"# {v} Prompt（以 {sample['id']} 为例；真实问题与证据按题替换）\n"]
        for m in msgs:
            md.append(f"## [{m['role']}]\n\n```text\n{m['content']}\n```\n")
        (pdir / f"{v}.md").write_text("\n".join(md), encoding="utf-8")
    (pdir / "fewshot_demos.json").write_text(json.dumps(DEMOS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("name", "n_items", "frozen_file_sha256", "fewshot_docs", "prompts")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
