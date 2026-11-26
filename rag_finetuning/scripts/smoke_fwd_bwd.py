"""GPU forward/backward smoke test: 4-bit base + LoRA on the LONGEST training sample.

Checks: finite loss, non-zero LoRA gradients, frozen base weights receive no gradient, peak memory. Prints JSON.
    python -m rag_finetuning.scripts.smoke_fwd_bwd --config rag_finetuning/configs/qlora_r8.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from rag_finetuning.common import REPO, read_jsonl
from rag_finetuning.scripts.train_qlora import encode, load_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    torch.manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    free0, total = torch.cuda.mem_get_info()
    tok, model = load_model(cfg)
    load_mib = torch.cuda.max_memory_allocated() / 2**20
    rows = read_jsonl(REPO / cfg["data"]["train"])
    enc = [encode(tok, r, cfg["data"]["max_seq_len"]) for r in rows]
    longest = max(enc, key=lambda x: len(x["input_ids"]))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"))
    model.train()
    batch = {k: torch.tensor([longest[k]], device="cuda") for k in ("input_ids", "attention_mask", "labels")}
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()
    torch.cuda.synchronize()
    dt = time.time() - t0
    lora_grads = [p.grad.float().norm().item() for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    base_with_grad = [n for n, p in model.named_parameters() if "lora_" not in n and p.grad is not None]
    out = {
        "longest_tokens": len(longest["input_ids"]), "completion_tokens": longest["n_completion"],
        "token_lengths": {"min": min(len(x["input_ids"]) for x in enc), "max": len(longest["input_ids"])},
        "loss": loss.item(), "loss_finite": math.isfinite(loss.item()),
        "lora_params_with_grad": len(lora_grads), "lora_grad_norm_sum": sum(lora_grads),
        "lora_grad_nonzero": sum(1 for g in lora_grads if g > 0), "base_params_with_grad": len(base_with_grad),
        "fwd_bwd_s": round(dt, 2), "free_before_mib": round(free0 / 2**20), "gpu_total_mib": round(total / 2**20),
        "peak_alloc_after_load_mib": round(load_mib), "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
    }
    out["ok"] = bool(out["loss_finite"] and out["lora_grad_nonzero"] > 0 and not base_with_grad)
    print("SMOKE", json.dumps(out))


if __name__ == "__main__":
    main()
