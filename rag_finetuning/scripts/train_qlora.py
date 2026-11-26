"""QLoRA SFT on the P6 RAG dataset (completion-only loss, platform prompt).

    python -m rag_finetuning.scripts.train_qlora --config rag_finetuning/configs/qlora_r8.yaml [--max-steps 10] [--tag smoke]

Writes to <output.root>/<run_name>[_<tag>]/ (outside Git): adapter/, train_log.jsonl, run_record.json.
The run record holds everything needed to reproduce / audit the run: model id + revision, dataset sha256 + counts,
LoRA / optimiser settings, seed, git commit, per-step loss / grad-norm / lr, eval loss, peak GPU memory, wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from rag_finetuning.common import REPO, messages_for, read_jsonl


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "rag_finetuning"], cwd=REPO, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_model(cfg: dict):
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=m["bnb_4bit_quant_type"],
                           bnb_4bit_use_double_quant=m["bnb_4bit_use_double_quant"],
                           bnb_4bit_compute_dtype=getattr(torch, m["compute_dtype"]))
    model = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], quantization_config=q,
                                                 device_map={"": 0}, dtype=getattr(torch, m["compute_dtype"]),
                                                 attn_implementation="sdpa")
    return tok, model


def encode(tok, rec: dict, max_len: int) -> dict:
    msgs = messages_for(rec)
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    # Qwen chat format: assistant turn ends with <|im_end|>; the model must learn to emit it (stop token).
    comp_ids = tok(rec["reference_answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = prompt_ids + comp_ids
    if len(ids) > max_len:
        raise ValueError(f"{rec['id']}: {len(ids)} tokens > max_seq_len {max_len} (refusing to truncate silently)")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + comp_ids,
            "n_prompt": len(prompt_ids), "n_completion": len(comp_ids)}


class Collator:
    def __init__(self, pad_id: int):
        self.pad = pad_id

    def __call__(self, feats):
        # pad to a multiple of 128 tokens: only a handful of distinct shapes → far less CUDA allocator fragmentation
        # (on Windows/WDDM fragmentation spills into shared system memory and slows training several-fold)
        L = max(len(f["input_ids"]) for f in feats)
        L = ((L + 127) // 128) * 128
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in feats:
            n = L - len(f["input_ids"])
            out["input_ids"].append(f["input_ids"] + [self.pad] * n)
            out["attention_mask"].append(f["attention_mask"] + [0] * n)
            out["labels"].append(f["labels"] + [-100] * n)
        return {k: torch.tensor(v) for k, v in out.items()}


class ListDS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {k: r[k] for k in ("input_ids", "attention_mask", "labels")}


class LogCB(TrainerCallback):
    def __init__(self, path: Path):
        self.path, self.t0 = path, time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        rec = dict(logs or {}, step=state.global_step, epoch=state.epoch, t=round(time.time() - self.t0, 1),
                   mem_alloc_peak_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                   mem_reserved_peak_mib=round(torch.cuda.max_memory_reserved() / 2**20))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps(rec, ensure_ascii=False), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    t = cfg["train"]
    if a.max_steps is not None:
        t["max_steps"] = a.max_steps
    if a.epochs is not None:
        t["num_epochs"] = a.epochs
    run = cfg["run_name"] + (f"_{a.tag}" if a.tag else "")
    out = Path(cfg["output"]["root"]) / run
    if (out / "adapter").exists():
        raise SystemExit(f"{out} already has an adapter; refusing to overwrite an experiment result (use another --tag)")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    set_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    tok, model = load_model(cfg)
    mem_after_load = torch.cuda.max_memory_allocated() / 2**20
    max_len = cfg["data"]["max_seq_len"]
    excl = set(cfg["data"].get("exclude_generation_methods") or [])
    tr_rows = [r for r in read_jsonl(REPO / cfg["data"]["train"]) if r["generation_method"] not in excl]
    va_rows = read_jsonl(REPO / cfg["data"]["val"])
    tr = [encode(tok, r, max_len) for r in tr_rows]
    va = [encode(tok, r, max_len) for r in va_rows]
    lens = sorted(len(x["input_ids"]) for x in tr)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=t["gradient_checkpointing"],
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler"], warmup_ratio=t["warmup_ratio"],
        num_train_epochs=t["num_epochs"], max_steps=t["max_steps"], weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"], optim=t["optim"], logging_steps=t["logging_steps"], bf16=True,
        eval_strategy="epoch" if t.get("eval_each_epoch") and t["max_steps"] in (-1, None) else "no",
        save_strategy="no", report_to=[], seed=cfg["seed"], data_seed=cfg["seed"], remove_unused_columns=False,
        gradient_checkpointing=t["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=ListDS(tr), eval_dataset=ListDS(va),
                      data_collator=Collator(tok.pad_token_id), callbacks=[LogCB(log_path)])

    pre_eval = trainer.evaluate()  # validation loss of the untrained adapter (= base model) for reference
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    post_eval = trainer.evaluate()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    logs = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    steps = [x for x in logs if "loss" in x]
    data_sha = {k: hashlib.sha256((REPO / cfg["data"][k]).read_bytes()).hexdigest() for k in ("train", "val")}
    record = {
        "run": run, "config_file": a.config, "config": cfg, "git_commit": git_commit(),
        "model": {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"], "quantization": "4-bit NF4 + double quant, bf16 compute"},
        "dataset": {"sha256": data_sha, "n_train": len(tr), "n_val": len(va), "excluded_generation_methods": sorted(excl),
                    "train_by_behavior": {b: sum(1 for r in tr_rows if r["behavior"] == b) for b in ("answer", "partial", "conflict", "refuse")},
                    "train_tokens": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]},
                    "completion_tokens_total": sum(x["n_completion"] for x in tr)},
        "trainable_params": trainable, "total_params": total,
        "steps": res.global_step, "epochs": res.metrics.get("epoch"), "train_loss_mean": res.training_loss,
        "first_step_loss": steps[0]["loss"] if steps else None, "last_step_loss": steps[-1]["loss"] if steps else None,
        "grad_norm_range": [min(x.get("grad_norm", math.inf) for x in steps), max(x.get("grad_norm", 0) for x in steps)] if steps else None,
        "eval_loss_before": pre_eval.get("eval_loss"), "eval_loss_after": post_eval.get("eval_loss"),
        "gpu": {"name": torch.cuda.get_device_name(0), "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                "allocated_after_model_load_mib": round(mem_after_load)},
        "wall_time_s": round(wall, 1), "adapter_dir": str(out / "adapter"),
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                     **{m: __import__(m).__version__ for m in ("transformers", "peft", "bitsandbytes", "accelerate")}},
        "python": sys.version.split()[0], "platform": os.name,
        "env": {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_OFFLINE")},
        "collator": "right-pad to multiple of 128 tokens",
    }
    (out / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_RECORD", json.dumps({k: record[k] for k in ("run", "steps", "first_step_loss", "last_step_loss",
                                                          "eval_loss_before", "eval_loss_after", "gpu", "wall_time_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
