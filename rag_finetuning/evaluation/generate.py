"""Generate answers for a built split with one model under FIXED conditions, and save them per question.

    # A: original model (same 4-bit NF4 base the adapter is trained on)
    python -m rag_finetuning.evaluation.generate --model base --split test --out <dir>
    # B: same base + QLoRA adapter
    python -m rag_finetuning.evaluation.generate --model base --adapter <run>/adapter --split test --out <dir>
    # reference only: local Ollama qwen2.5:7b (different size: NOT evidence of a fine-tuning effect)
    python -m rag_finetuning.evaluation.generate --model ollama:qwen2.5:7b --split test --out <dir>

Fixed for every model: same records (evidence + order), same platform prompt (rag_v1.0), greedy decoding,
max_new_tokens, batch size 1. Records per question: answer, latency, prompt/output tokens, GPU peak memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from rag_finetuning.common import BUILT, PROMPT_VERSION, messages_for, prompt_hash, read_jsonl, write_jsonl

GEN = {"max_new_tokens": 384, "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
       "repetition_penalty": 1.0}
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


class HFGenerator:
    def __init__(self, adapter: str | None = None, quant: str = "nf4"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        kw = {"device_map": {"": 0}, "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if quant == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_use_double_quant=True,
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, **kw)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self.load_peak_mib = round(torch.cuda.max_memory_allocated() / 2**20)
        self.name = f"qwen2.5-1.5b-instruct[{quant}]" + (f"+lora:{Path(adapter).parent.name}" if adapter else "")

    def __call__(self, messages):
        torch = self.torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=self.tok.eos_token_id,
                                      **{k: v for k, v in GEN.items() if v is not None})
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        new = out[0, ids.shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return {"answer": text, "latency_ms": round(dt, 1), "prompt_tokens": int(ids.shape[1]),
                "output_tokens": int(new.shape[0]), "hit_max_tokens": int(new.shape[0]) >= GEN["max_new_tokens"],
                "gpu_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20)}


class OllamaGenerator:
    def __init__(self, model: str):
        from rag_finetuning.compat.local_llm import OllamaClient
        self.c = OllamaClient(model=model, timeout_s=600)
        self.info = self.c.model_info()          # raises model_missing instead of downloading
        self.name = f"ollama:{model}"
        self.load_peak_mib = None

    def __call__(self, messages):
        from rag_finetuning.compat.local_llm import gpu_memory_used_mib
        g = self.c.chat(messages, {"temperature": 0, "seed": 42, "num_predict": GEN["max_new_tokens"], "num_ctx": 4096})
        return {"answer": g.text, "latency_ms": g.total_ms, "prompt_tokens": g.prompt_tokens,
                "output_tokens": g.output_tokens, "hit_max_tokens": g.done_reason == "length",
                "gpu_used_total_mib": gpu_memory_used_mib()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base | ollama:<name>")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--quant", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=None, help="explicit jsonl (default data/built/<split>.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    path = Path(a.data) if a.data else BUILT / f"{a.split}.jsonl"
    rows = read_jsonl(path)[: a.limit]
    gen = OllamaGenerator(a.model.split(":", 1)[1]) if a.model.startswith("ollama:") else HFGenerator(a.adapter, a.quant)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        g = gen(messages_for(r))
        res.append({"id": r["id"], "model": gen.name, **g})
        print(f"[{i}/{len(rows)}] {r['id']} {g['latency_ms']:.0f}ms {g['output_tokens']}tok", flush=True)
    write_jsonl(out_dir / "generations.jsonl", res)
    meta = {"model": gen.name, "adapter": a.adapter, "quant": a.quant if not a.model.startswith("ollama:") else None,
            "data": str(path), "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": len(res),
            "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_hash(), "generation": GEN,
            "load_peak_alloc_mib": gen.load_peak_mib, "wall_s": round(time.time() - t0, 1),
            "ollama_info": getattr(gen, "info", None)}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", json.dumps({k: meta[k] for k in ("model", "n", "wall_s", "load_peak_alloc_mib")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
