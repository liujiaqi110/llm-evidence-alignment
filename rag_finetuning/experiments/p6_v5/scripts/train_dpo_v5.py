"""P6-v5 实验二：QLoRA DPO（真正的偏好优化，TRL DPOTrainer，sigmoid 损失）。

算法与参考策略（按 TRL 0.23 的 PEFT 双适配器用法，不是 SFT，也不是 reference-free）：
* 底座：Qwen2.5-3B-Instruct 4-bit NF4（与 SFT 相同）；
* 策略 π_θ：加载 SFT 选中的适配器为可训练适配器 "policy"；
* 参考 π_ref：同一 SFT 适配器再加载一份为冻结适配器 "reference"（DPOConfig.ref_adapter_name）。
  TRL 计算参考 logp 时切换到 "reference" 适配器、no_grad；因此 π_ref = SFT 模型，与标准 DPO 定义一致（参考 = SFT 初始策略）。
* 损失：-log σ(β[(log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x))])，β = 0.1。

    # 显存冒烟（只跑若干步，记录峰值，不保存适配器）
    P6_MEM_FRACTION=0.74 python -m rag_finetuning.experiments.p6_v5.scripts.train_dpo_v5 --config <yaml> --smoke 10
    # 正式训练 + 在 hard_dev 上选择检查点
    P6_MEM_FRACTION=0.74 python -m rag_finetuning.experiments.p6_v5.scripts.train_dpo_v5 --config <yaml>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

from rag_finetuning.common import REPO, read_jsonl
from rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 import msgs


def build_dataset(path: Path, seed: int):
    from datasets import Dataset
    rows = read_jsonl(path)
    random.Random(seed).shuffle(rows)
    data = [{"prompt": msgs(r), "chosen": [{"role": "assistant", "content": r["chosen"]}],
             "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows]
    return Dataset.from_list(data), rows


def make_step_cb(path: Path):
    """逐步记录显存与耗时（2026-11-25 正式 P42/P43 前加入；只记录，不影响训练）。
    每个优化步写一行到 step_stats.jsonl（即时写盘，OOM 时也保留）；同时把 Trainer 日志即时写到 train_log_live.jsonl。"""
    from transformers import TrainerCallback

    class StepStats(TrainerCallback):
        def __init__(self):
            self.t = None
            self.max_alloc = 0
            self.max_reserved = 0
            path.mkdir(parents=True, exist_ok=True)
            self.f = open(path / "step_stats.jsonl", "a", encoding="utf-8")
            self.flog = open(path / "train_log_live.jsonl", "a", encoding="utf-8")

        def on_step_begin(self, args, state, control, **kw):
            if self.t is None:
                self.t = time.time()
                torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            a, r = torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20
            free, total = torch.cuda.mem_get_info()
            self.max_alloc, self.max_reserved = max(self.max_alloc, a), max(self.max_reserved, r)
            self.f.write(json.dumps({"step": state.global_step, "step_time_s": round(now - self.t, 2), "peak_allocated_mib": round(a),
                                     "peak_reserved_mib": round(r), "allocated_end_mib": round(torch.cuda.memory_allocated() / 2**20),
                                     "device_free_mib": round(free / 2**20), "time": time.strftime("%F %T")}) + "\n")
            self.f.flush()
            self.t = time.time()
            torch.cuda.reset_peak_memory_stats()

        def on_log(self, args, state, control, logs=None, **kw):
            self.flog.write(json.dumps({"step": state.global_step, **(logs or {})}, ensure_ascii=False) + "\n")
            self.flog.flush()

    return StepStats()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--logits-to-keep", action="store_true", help="只对回答 token 计算 logits（不改变损失）")
    ap.add_argument("--precompute-ref", action="store_true", help="训练前预先计算参考 logp（不改变损失）")
    ap.add_argument("--tag", default="", help="冒烟测试输出目录后缀，避免覆盖")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    frac = os.environ.get("P6_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer
    import peft, transformers, trl
    set_seed(cfg["seed"])
    mid, rev = cfg["model"]["id"], cfg["model"]["revision"]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(mid, revision=rev, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16,
                                                local_files_only=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    sft = cfg["init_adapter"]
    model = PeftModel.from_pretrained(base, sft, adapter_name="policy", is_trainable=True)
    model.load_adapter(sft, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    load_peak = round(torch.cuda.max_memory_allocated() / 2**20)
    ds, rows = build_dataset(REPO / cfg["data"]["pairs"], cfg["seed"])
    if a.smoke:
        ds = ds.select(range(min(len(ds), a.smoke * cfg["train"]["gradient_accumulation_steps"])))
    out = Path(cfg["output"]["root"]) / (cfg["run_name"] + (f"_smoke{a.tag}" if a.smoke else ""))
    if out.exists() and not a.smoke:
        raise SystemExit(f"{out} 已存在：不覆盖已有训练结果")
    steps_per_epoch = max(1, len(ds) // cfg["train"]["gradient_accumulation_steps"])
    max_len = a.max_length or cfg["data"]["max_length"]
    dcfg = DPOConfig(
        output_dir=str(out), model_adapter_name="policy", ref_adapter_name="reference", beta=cfg["train"]["beta"], loss_type="sigmoid",
        learning_rate=cfg["train"]["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.1,
        num_train_epochs=cfg["train"]["num_epochs"] if not a.smoke else 1, max_steps=a.smoke if a.smoke else -1,
        per_device_train_batch_size=1, gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        max_length=max_len, max_prompt_length=min(cfg["data"]["max_prompt_length"], max_len - 128),
        use_logits_to_keep=a.logits_to_keep or cfg["train"].get("use_logits_to_keep", False),
        precompute_ref_log_probs=a.precompute_ref or cfg["train"].get("precompute_ref_log_probs", False),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True, optim="paged_adamw_8bit",
        logging_steps=1, save_strategy="no" if a.smoke else "steps", save_steps=max(1, steps_per_epoch // 2), save_only_model=True,
        report_to=[], seed=cfg["seed"], remove_unused_columns=False, dataloader_num_workers=0)
    step_cb = make_step_cb(out)
    trainer = DPOTrainer(model=model, ref_model=None, args=dcfg, train_dataset=ds, processing_class=tok, callbacks=[step_cb])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ok, err = True, None
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        ok, err = False, f"CUDA OOM: {str(e)[:300]}"
    wall = round(time.time() - t0, 1)
    peak = {"peak_allocated_mib": round(max(step_cb.max_alloc, torch.cuda.max_memory_allocated() / 2**20)),
            "peak_reserved_mib": round(max(step_cb.max_reserved, torch.cuda.max_memory_reserved() / 2**20)),
            "load_peak_mib": load_peak, "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20)}
    out.mkdir(parents=True, exist_ok=True)
    logs = trainer.state.log_history
    with open(out / "train_log.jsonl", "w", encoding="utf-8") as f:
        for x in logs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    from rag_finetuning.scripts.train_qlora import git_commit
    rec = {"run": cfg["run_name"], "git_commit": git_commit(), "config_file": a.config, "smoke": a.smoke, "ok": ok, "error": err, "config": cfg, "max_length_used": max_len, "use_logits_to_keep": dcfg.use_logits_to_keep, "precompute_ref_log_probs": dcfg.precompute_ref_log_probs,
           "algorithm": "DPO sigmoid（TRL DPOTrainer），policy=SFT 适配器可训练副本，reference=同一 SFT 适配器冻结副本",
           "init_adapter": sft, "init_adapter_sha256": hashlib.sha256((Path(sft) / "adapter_model.safetensors").read_bytes()).hexdigest(),
           "pairs_file": cfg["data"]["pairs"], "pairs_sha256": hashlib.sha256((REPO / cfg["data"]["pairs"]).read_bytes()).hexdigest(),
           "n_pairs_used": len(ds), "steps": trainer.state.global_step, "trainable_params": n_trainable, "wall_time_s": wall, "gpu": peak,
           "has_preference_metrics": any("rewards/margins" in x for x in logs),
           "versions": {"trl": trl.__version__, "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__},
           "first_logs": [x for x in logs if "loss" in x][:3], "last_logs": [x for x in logs if "loss" in x][-3:]}
    (out / "run_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN", json.dumps({k: rec[k] for k in ("ok", "error", "steps", "wall_time_s", "gpu", "has_preference_metrics")}, ensure_ascii=False), flush=True)
    if a.smoke or not ok:
        return 0 if ok else 2
    # ---------------- 在 hard_dev 上选择检查点（只用 "policy" 适配器）
    del trainer, model, base
    torch.cuda.empty_cache()
    from rag_finetuning.experiments.p6_v4.scripts.train_v4 import load
    from rag_finetuning.experiments.p6_v5.scripts.train_sft_v5 import dev_scores_v5
    dev = read_jsonl(REPO / cfg["data"]["dev"])
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    tok2, base2 = load(None)
    m2, res = None, []
    cand = [("init_sft", Path(sft))] + [(c.name, c / "policy" if (c / "policy").exists() else c) for c in ckpts]
    for name, path in cand:
        an = name.replace("-", "_")
        if m2 is None:
            m2 = PeftModel.from_pretrained(base2, str(path), adapter_name=an)
        else:
            m2.load_adapter(str(path), adapter_name=an)
        m2.set_adapter(an)
        m2.eval()
        agg = dev_scores_v5(m2, tok2, dev, save_path=out / "dev_eval" / f"{name}.jsonl")
        agg["checkpoint"] = name
        res.append(agg)
        print("DEV", agg, flush=True)
    dpo_res = [r for r in res if r["checkpoint"] != "init_sft"]
    best = sorted(dpo_res, key=lambda x: (-x["dev_multi_v5"], -x["dev_content_v4"], x["dev_whole_refusal"],
                                          int(x["checkpoint"].split("-")[1])))[0]
    src = dict(cand)[best["checkpoint"]]
    shutil.copytree(src, out / "adapter")
    rec["checkpoint_selection"] = {"rule": "max hard_dev multi_complete_v5 → max content_ok_v4 → min whole_refusal → earliest（init_sft 只作参照，不参与选择）",
                                   "per_checkpoint": res, "selected": best["checkpoint"]}
    rec["adapter_sha256"] = hashlib.sha256((out / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    (out / "run_record.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SELECTED", best["checkpoint"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
