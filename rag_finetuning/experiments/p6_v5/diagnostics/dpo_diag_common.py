"""DPO 诊断公用：与 scripts/train_dpo_v5.py 完全相同的模型加载与 DPOConfig（只读复用，不改正式训练脚本）。"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from rag_finetuning.common import REPO
from rag_finetuning.experiments.p6_v5.scripts.train_dpo_v5 import build_dataset

V5 = Path(__file__).resolve().parents[1]
PAIRS = REPO / "rag_finetuning/experiments/p6_v5/data/train_pref/dpo_pairs.jsonl"
MID, REV = "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INIT = "workspace/runs/p6_v4/v4_3b_s42/adapter"   # 与冒烟测试相同（正式 SFT 适配器尚未训练）
MAX_LEN, MAX_PROMPT = 1024, 896


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MID, revision=REV, local_files_only=True)


def tokenized_pairs(tok, max_prompt_length=MAX_PROMPT):
    """按 TRL 0.23.1 DPOTrainer._prepare_dataset 的同一流程分词（maybe_extract_prompt → maybe_apply_chat_template → tokenize_row）。
    返回 (rows, tokenized)，顺序与 build_dataset（种子 42 打乱）一致。"""
    from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
    from trl import DPOTrainer
    ds, rows = build_dataset(PAIRS, 42)
    out = []
    for ex in ds:
        ex = maybe_extract_prompt(ex)
        ex = {**ex, **maybe_apply_chat_template(ex, tok)}
        out.append(DPOTrainer.tokenize_row(ex, tok, max_prompt_length=max_prompt_length, max_completion_length=None, add_special_tokens=False))
    return rows, out


def load_policy_ref():
    frac = os.environ.get("P6_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(MID, revision=REV, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16, local_files_only=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, INIT, adapter_name="policy", is_trainable=True)
    model.load_adapter(INIT, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    return model


def dpo_config(out, **over):
    from trl import DPOConfig
    kw = dict(output_dir=str(out), model_adapter_name="policy", ref_adapter_name="reference", beta=0.1, loss_type="sigmoid",
              learning_rate=2e-5, lr_scheduler_type="cosine", warmup_ratio=0.1, num_train_epochs=1,
              per_device_train_batch_size=1, gradient_accumulation_steps=8, max_length=MAX_LEN, max_prompt_length=MAX_PROMPT,
              use_logits_to_keep=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
              bf16=True, optim="paged_adamw_8bit", logging_steps=1, save_strategy="no", save_only_model=True,
              report_to=[], seed=42, remove_unused_columns=False, dataloader_num_workers=0)
    kw.update(over)
    return DPOConfig(**kw)
