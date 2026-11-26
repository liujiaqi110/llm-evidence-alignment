"""P6-v3 QLoRA 训练：复用 P6-v1 的已审查管线（train_qlora.encode / Collator / LogCB），增加
  * 每个 epoch 结束保存一次 Adapter（checkpoint_epoch{k}/）；
  * 预先注册的停止点选择：在开发集（dev，文档与训练 / 测试均不重叠）上生成并用评分器 v2 打分，
    选 content_ok 最高的 epoch；平分 → 错误拒答更少；再平分 → 更早的 epoch。测试集不参与。

    python -m rag_finetuning.experiments.p6_v3.scripts.train_v3 --config rag_finetuning/experiments/p6_v3/configs/<cfg>.yaml
    python -m rag_finetuning.experiments.p6_v3.scripts.train_v3 --config ... --smoke 10      # 冒烟：10 步，不选 epoch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from rag_finetuning.common import REPO, read_jsonl
from rag_finetuning.scripts.train_qlora import Collator, ListDS, LogCB, encode, git_commit, load_model


class EpochSaver(TrainerCallback):
    def __init__(self, out: Path):
        self.out = out

    def on_epoch_end(self, args, state, control, model=None, **kw):
        k = int(round(state.epoch))
        d = self.out / f"checkpoint_epoch{k}"
        model.save_pretrained(d)
        (d / "epoch.json").write_text(json.dumps({"epoch": k, "global_step": state.global_step}), encoding="utf-8")


def select_epoch(cfg: dict, out: Path, tok_dir: Path) -> dict:
    """在 dev 上逐 epoch 生成并评分（评分器 v2），按预注册规则选择。"""
    from rag_finetuning.evaluation.generate import GEN
    from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import score_item_v2
    from rag_finetuning.common import messages_for
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    dev = read_jsonl(REPO / cfg["data"]["dev"])
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(m["id"], revision=m["revision"], quantization_config=q, device_map={"": 0},
                                                dtype=torch.bfloat16)
    res = []
    ckpts = sorted(out.glob("checkpoint_epoch*"), key=lambda p: int(p.name.replace("checkpoint_epoch", "")))
    model = None
    for ck in ckpts:
        if model is None:
            model = PeftModel.from_pretrained(base, str(ck), adapter_name=ck.name)
        else:
            model.load_adapter(str(ck), adapter_name=ck.name)
        model.set_adapter(ck.name)
        model.eval()
        ok = fr = 0
        for r in dev:
            ids = tok.apply_chat_template(messages_for(r), add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id,
                                   **{k: v for k, v in GEN.items() if v is not None})
            ans = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()
            s = score_item_v2(r, ans)
            ok += int(s["content_ok"])
            fr += int(s["false_refusal"])
        res.append({"epoch": int(ck.name.replace("checkpoint_epoch", "")), "dev_content_ok": ok, "dev_false_refusal": fr,
                    "dev_n": len(dev)})
        print("DEV", res[-1], flush=True)
    best = sorted(res, key=lambda x: (-x["dev_content_ok"], x["dev_false_refusal"], x["epoch"]))[0]
    return {"rule": "max dev content_ok → min dev false_refusal → earliest epoch", "per_epoch": res, "selected_epoch": best["epoch"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None, help="覆盖 num_epochs（消融用主模型选出的 epoch 数）")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    t = cfg["train"]
    if a.epochs is not None:
        t["num_epochs"] = a.epochs
    run = cfg["run_name"] + (f"_smoke{a.smoke}" if a.smoke else "")
    out = Path(cfg["output"]["root"]) / run
    if (out / "adapter").exists():
        raise SystemExit(f"{out} 已有 adapter，拒绝覆盖已有实验结果")
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    set_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    tok, model = load_model(cfg)
    after_load = torch.cuda.max_memory_allocated() / 2**20
    max_len = cfg["data"]["max_seq_len"]
    tr_rows = read_jsonl(REPO / cfg["data"]["train"])
    dev_rows = read_jsonl(REPO / cfg["data"]["dev"])
    tr = [encode(tok, r, max_len) for r in tr_rows]
    dv = [encode(tok, r, max_len) for r in dev_rows]
    lens = sorted(len(x["input_ids"]) for x in tr)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args = TrainingArguments(
        output_dir=str(out / "hf_tmp"), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"], warmup_ratio=t["warmup_ratio"], num_train_epochs=t["num_epochs"],
        max_steps=a.smoke or -1, weight_decay=0.0, max_grad_norm=1.0, optim=t["optim"], logging_steps=1, bf16=True,
        eval_strategy="no" if a.smoke else "epoch", save_strategy="no", report_to=[], seed=cfg["seed"],
        data_seed=cfg["seed"], remove_unused_columns=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=0)
    cbs = [LogCB(log_path)] + ([] if a.smoke else [EpochSaver(out)])
    trainer = Trainer(model=model, args=args, train_dataset=ListDS(tr), eval_dataset=ListDS(dv),
                      data_collator=Collator(tok.pad_token_id), callbacks=cbs)
    pre = trainer.evaluate() if not a.smoke else {}
    t0 = time.time()
    res = trainer.train()
    wall = time.time() - t0
    peak_alloc, peak_res = torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20
    logs = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    steps = [x for x in logs if "loss" in x]
    rec = {"run": run, "config": cfg, "config_file": a.config, "git_commit": git_commit(),
           "model": {"id": cfg["model"]["id"], "revision": cfg["model"]["revision"], "quantization": "4-bit NF4 + double quant, bf16 compute"},
           "dataset": {"train_file": cfg["data"]["train"], "train_sha256": hashlib.sha256((REPO / cfg["data"]["train"]).read_bytes()).hexdigest(),
                       "dev_sha256": hashlib.sha256((REPO / cfg["data"]["dev"]).read_bytes()).hexdigest(),
                       "n_train": len(tr), "n_dev": len(dv), "train_tokens": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]},
                       "completion_tokens_total": sum(x["n_completion"] for x in tr)},
           "trainable_params": trainable, "steps": res.global_step, "epochs": res.metrics.get("epoch"),
           "train_loss_mean": res.training_loss, "first_step_loss": steps[0]["loss"] if steps else None,
           "last_step_loss": steps[-1]["loss"] if steps else None,
           "grad_norm_range": [min(x["grad_norm"] for x in steps), max(x["grad_norm"] for x in steps)] if steps else None,
           "dev_loss_before": pre.get("eval_loss"), "dev_loss_by_epoch": [x["eval_loss"] for x in logs if "eval_loss" in x],
           "gpu": {"name": torch.cuda.get_device_name(0), "peak_allocated_mib": round(peak_alloc), "peak_reserved_mib": round(peak_res),
                   "allocated_after_model_load_mib": round(after_load),
                   "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20)},
           "wall_time_s": round(wall, 1), "env": {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_OFFLINE")},
           "versions": {"torch": torch.__version__, **{m: __import__(m).__version__ for m in ("transformers", "peft", "bitsandbytes")}}}
    if a.smoke:
        model.save_pretrained(out / "adapter_smoke")
    else:
        del trainer, model
        torch.cuda.empty_cache()
        sel = select_epoch(cfg, out, out)
        rec["epoch_selection"] = sel
        shutil.copytree(out / f"checkpoint_epoch{sel['selected_epoch']}", out / "adapter")
    rec["adapter_dir"] = str(out / ("adapter_smoke" if a.smoke else "adapter"))
    (out / "run_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN_RECORD", json.dumps({k: rec[k] for k in ("run", "steps", "first_step_loss", "last_step_loss", "gpu", "wall_time_s")}
                                   | {"dev_loss": rec["dev_loss_by_epoch"], "sel": rec.get("epoch_selection", {}).get("selected_epoch")},
                                   ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
