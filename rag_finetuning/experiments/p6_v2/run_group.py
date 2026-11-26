"""在冻结输入上运行一个实验组，逐题保存答案与成本。

    python -m rag_finetuning.experiments.p6_v2.run_group --group A --input diagnostic_47 --out <dir>

实验组（见 config.yaml）：
  A  Qwen2.5-1.5B-Instruct（4-bit NF4）            zero_shot
  B  同上 + P6-v1 QLoRA Adapter（qlora_r8_v1）     zero_shot
  C  Ollama qwen2.5:7b（Q4_K_M）                  zero_shot
  D  Ollama qwen2.5:7b（Q4_K_M）                  few_shot
  E  Qwen2.5-1.5B-Instruct（4-bit NF4）            few_shot   （额外参考组）

生成设置：贪心 / temperature 0，最多 384 个输出 token，batch 1；Ollama 另设 seed 42、num_ctx 8192。
Token 口径：
  input_tokens_hf      —— 用 HF Qwen2.5 分词器对完整 chat 模板计数，所有组同一口径（成本对照用这一列）；
  ollama_prompt_eval_count —— Ollama 报告的实际计算的 prompt token；有前缀缓存时会小于完整长度，不可与 HF 口径混比；
  output_tokens        —— HF 为生成 token 数，Ollama 为 eval_count（同一分词器族，口径接近但不保证完全一致）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

from rag_finetuning.common import read_jsonl, write_jsonl
from rag_finetuning.evaluation.generate import GEN, MODEL_ID, REVISION, HFGenerator
from rag_finetuning.experiments.p6_v2.prompts import messages, prompt_fingerprint
from rag_finetuning.compat.rag_pipeline import is_refusal_text, validate_citations

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


class OllamaGen:
    def __init__(self, model: str, opts: dict):
        from rag_finetuning.compat.local_llm import OllamaClient
        self.c = OllamaClient(model=model, timeout_s=900)
        self.info = self.c.model_info()
        self.opts = opts
        self.name = f"ollama:{model}"

    def __call__(self, msgs):
        from rag_finetuning.compat.local_llm import gpu_memory_used_mib
        g = self.c.chat(msgs, dict(self.opts))
        return {"answer": g.text, "latency_ms": g.total_ms, "ollama_prompt_eval_count": g.prompt_tokens,
                "output_tokens": g.output_tokens, "hit_max_tokens": g.done_reason == "length",
                "ollama_total_ms": g.ollama_total_ms, "ollama_prompt_eval_ms": g.prompt_eval_ms,
                "ollama_eval_ms": g.eval_ms, "gpu_used_total_mib": gpu_memory_used_mib()}


def status_of(answer: str, ctx_ids):
    cited, valid, blocked, _ = validate_citations(answer, ctx_ids)
    if is_refusal_text(answer):
        st = "refused"
    elif not valid:
        st = "uncited"
    else:
        st = "ok"
    return st, cited, valid, blocked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--input", required=True, help="frozen_inputs/<name>.jsonl 的名称")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    g = CFG["groups"][a.group]
    path = HERE / "frozen_inputs" / f"{a.input}.jsonl"
    manifest = json.loads((HERE / "frozen_inputs" / f"{a.input}.manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["frozen_file_sha256"]:
        raise SystemExit("冻结输入的哈希与 manifest 不一致：拒绝运行")
    rows = read_jsonl(path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    if g["backend"] == "hf":
        gen = HFGenerator(adapter=g.get("adapter"), quant="nf4")
        settings = {"backend": "transformers", **{k: v for k, v in GEN.items() if v is not None}, "batch_size": 1,
                    "quantization": "bitsandbytes 4-bit NF4 + double quant, bf16 compute",
                    "model_id": MODEL_ID, "revision": REVISION, "adapter": g.get("adapter")}
    else:
        opts = dict(CFG["ollama_options"])
        gen = OllamaGen(g["model"], opts)
        settings = {"backend": "ollama", **opts, "model": g["model"], "ollama_model_info": gen.info}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        msgs = messages(r, g["prompt_variant"])
        n_in = len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
        o = gen(msgs)
        ctx = [e["cid"] for e in r["evidence"]]
        st, cited, valid, blocked = status_of(o["answer"], ctx)
        res.append({"question_id": r["id"], "id": r["id"], "question": r["question"], "evidence_ids": ctx,
                    "input_sha256": r["input_sha256"], "group": a.group, "model": gen.name,
                    "prompt_variant": g["prompt_variant"], "generated_answer": o["answer"], "answer": o["answer"],
                    "citations": cited, "valid_citations": valid, "invalid_citations": blocked, "status": st,
                    "input_tokens_hf": n_in, **{k: v for k, v in o.items() if k != "answer"},
                    "generation_settings": settings})
        print(f"[{i}/{len(rows)}] {r['id']} {st} in={n_in} out={o.get('output_tokens')} {o['latency_ms']:.0f}ms", flush=True)
    write_jsonl(out / "generations.jsonl", res)
    meta = {"group": a.group, "model": gen.name, "prompt_variant": g["prompt_variant"], "input": a.input,
            "frozen_file_sha256": manifest["frozen_file_sha256"], "prompts": prompt_fingerprint(),
            "generation_settings": settings, "n": len(res), "wall_s": round(time.time() - t0, 1),
            "load_peak_alloc_mib": getattr(gen, "load_peak_mib", None)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if g["backend"] == "ollama":   # 释放显存，避免影响后续 HF 组
        import urllib.request
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", method="POST",
                               data=json.dumps({"model": g["model"], "keep_alive": 0}).encode()), timeout=60).read()
    print("DONE", json.dumps({k: meta[k] for k in ("group", "model", "prompt_variant", "n", "wall_s")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
