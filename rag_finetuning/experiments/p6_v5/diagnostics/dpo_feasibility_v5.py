"""DPO 训练可行性复测（2026-11-25；不是正式训练，生成的临时权重不得用于正式训练或任何评测）。

    # 1) 全量预检（CPU）：580 对按 TRL 0.23.1 的实际分词流程检查截断 / chosen-rejected prompt 一致 / 回答缺失，选出最坏情况集合
    python -m rag_finetuning.experiments.p6_v5.diagnostics.dpo_feasibility_v5 precheck
    # 2) 最坏情况显存复测（GPU）：policy = S4x epoch3 适配器（可训练），reference = 同一适配器冻结副本
    P6_MEM_FRACTION=0.74 python -m rag_finetuning.experiments.p6_v5.diagnostics.dpo_feasibility_v5 memory --seed 42

memory：数据 = 最长序列 8 对 + 最长回答 8 对 + 最长序列 8 对（共 3 个优化步，每步 8 对 × 梯度累积 8），
DPO 超参数与冻结配置完全相同（configs/dpo_3b_s4x.yaml：beta 0.1、lr 2e-5 cosine warmup 0.1、1024 / 896、use_logits_to_keep=True、
paged_adamw_8bit、梯度检查点）；save_steps=1 测试检查点保存。
同时检查：DPOTrainer 实际分词结果与预检一致；reference 参数全程不变且 requires_grad=False；只有 policy 参数被更新。
输出：diagnostics/dpo_feasibility/（precheck.json、memory_s4x.json）；权重副本在 runs/p6_v5/diag_feasibility_s4x/（标记 DO_NOT_USE）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import yaml

from rag_finetuning.common import REPO

V5 = Path(__file__).resolve().parents[1]
OUT = V5 / "diagnostics" / "dpo_feasibility"
RUNS = Path("workspace/runs/p6_v5")
FROZEN_PAIRS_SHA = "61804f6636b81277f2482550713ab201fc8c10a6ee14be2329280b94ca448d06"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cfg_for(seed: int) -> dict:
    return yaml.safe_load((V5 / f"configs/dpo_3b_s{seed}.yaml").read_text(encoding="utf-8"))


def tokenize_all(tok, cfg, max_prompt_length):
    from trl import DPOTrainer
    from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt
    from rag_finetuning.experiments.p6_v5.scripts.train_dpo_v5 import build_dataset
    ds, rows = build_dataset(REPO / cfg["data"]["pairs"], cfg["seed"])
    out, texts = [], []
    for ex in ds:
        ex = maybe_extract_prompt(ex)
        t = {**ex, **maybe_apply_chat_template(ex, tok)}
        texts.append(t)
        out.append(DPOTrainer.tokenize_row(t, tok, max_prompt_length=max_prompt_length, max_completion_length=None, add_special_tokens=False))
    return rows, out, texts


def precheck() -> int:
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = cfg_for(42)
    assert sha(REPO / cfg["data"]["pairs"]) == FROZEN_PAIRS_SHA
    tok = AutoTokenizer.from_pretrained(cfg["model"]["id"], revision=cfg["model"]["revision"], local_files_only=True)
    L, LP = cfg["data"]["max_length"], cfg["data"]["max_prompt_length"]
    rows, tk, texts = tokenize_all(tok, cfg, LP)
    _, tk_full, _ = tokenize_all(tok, cfg, None)
    eos = tok.eos_token_id
    info, problems = [], []
    for r, t, tf, tx in zip(rows, tk, tk_full, texts):
        p, c, j = len(t["prompt_input_ids"]), len(t["chosen_input_ids"]), len(t["rejected_input_ids"])
        x = {"pair_id": r["pair_id"], "prompt": p, "prompt_untruncated": len(tf["prompt_input_ids"]), "chosen": c, "rejected": j,
             "seq_chosen": p + c, "seq_rejected": p + j}
        issues = []
        if x["prompt_untruncated"] > LP:
            issues.append("prompt 超过 max_prompt_length，被左截断")
        if max(x["seq_chosen"], x["seq_rejected"]) > L:
            issues.append("[prompt + 回答] 超过 max_length，会被截断（chosen / rejected 看到的 prompt 可能不同）")
        if not r["chosen"].strip() or c <= 1:
            issues.append("chosen 回答缺失")
        if not r["rejected"].strip() or j <= 1:
            issues.append("rejected 回答缺失")
        if t["chosen_input_ids"][-1] != eos or t["rejected_input_ids"][-1] != eos:
            issues.append("回答末尾缺少 eos")
        if r["chosen"].strip() == r["rejected"].strip():
            issues.append("chosen 与 rejected 相同")
        if not tx["prompt"].rstrip().endswith("<|im_start|>assistant"):
            issues.append("prompt 未以 assistant 生成前缀结尾（chat template 异常）")
        if r["chosen"].strip() not in tok.decode(t["chosen_input_ids"]) or r["rejected"].strip() not in tok.decode(t["rejected_input_ids"]):
            issues.append("回答分词后解码与原文不一致")
        x["issues"] = issues
        if issues:
            problems.append(x)
        info.append(x)
    by_seq = sorted(info, key=lambda x: (-max(x["seq_chosen"], x["seq_rejected"]), x["pair_id"]))
    by_comp = sorted(info, key=lambda x: (-max(x["chosen"], x["rejected"]), x["pair_id"]))
    by_sum = sorted(info, key=lambda x: (-(x["seq_chosen"] + x["seq_rejected"]), x["pair_id"]))
    sets = {"longest_seq_8": [x["pair_id"] for x in by_seq[:8]], "longest_completion_8": [x["pair_id"] for x in by_comp[:8]],
            "longest_seq_sum_8": [x["pair_id"] for x in by_sum[:8]]}
    seq = [max(x["seq_chosen"], x["seq_rejected"]) for x in info]
    rep = {"when": time.strftime("%F %T"), "pairs_file": cfg["data"]["pairs"], "pairs_sha256": FROZEN_PAIRS_SHA, "n_pairs": len(info),
           "tokenizer": f'{cfg["model"]["id"]}@{cfg["model"]["revision"]}', "flow": "TRL 0.23.1：maybe_extract_prompt → maybe_apply_chat_template → DPOTrainer.tokenize_row",
           "max_length": L, "max_prompt_length": LP,
           "prompt_over_max_prompt": sum(x["prompt_untruncated"] > LP for x in info),
           "pairs_over_max_length": sum(max(x["seq_chosen"], x["seq_rejected"]) > L for x in info),
           "pairs_with_any_issue": len(problems), "issues": problems,
           "seq_len": {"median": statistics.median(seq), "p95": sorted(seq)[int(.95 * len(seq))], "max": max(seq)},
           "prompt_len_max": max(x["prompt"] for x in info), "completion_len_max": max(max(x["chosen"], x["rejected"]) for x in info),
           "worst_case_sets": sets,
           "worst_case_detail": {k: [next(x for x in info if x["pair_id"] == i) for i in v] for k, v in sets.items()}}
    (OUT / "precheck.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("worst_case_detail", "issues")}, ensure_ascii=False, indent=1))
    return 0 if not problems else 3


def memory(seed: int) -> int:
    from peft import PeftModel, prepare_model_for_kbit_training
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer
    import peft, transformers, trl
    from rag_finetuning.experiments.p6_v5.scripts.train_dpo_v5 import build_dataset

    cfg = cfg_for(seed)
    pre = json.loads((OUT / "precheck.json").read_text(encoding="utf-8"))
    assert pre["pairs_with_any_issue"] == 0 and sha(REPO / cfg["data"]["pairs"]) == FROZEN_PAIRS_SHA
    frac = float(os.environ.get("P6_MEM_FRACTION", "1"))
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    free0, total = torch.cuda.mem_get_info()
    set_seed(cfg["seed"])
    init = Path(cfg["init_adapter"])
    init_sha = sha(init / "adapter_model.safetensors")
    mid, rev = cfg["model"]["id"], cfg["model"]["revision"]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(mid, revision=rev, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16, local_files_only=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base, str(init), adapter_name="policy", is_trainable=True)
    model.load_adapter(str(init), adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    load_peak = round(torch.cuda.max_memory_allocated() / 2**20)
    named = dict(model.named_parameters())
    ref_names = [n for n in named if ".reference." in n]
    pol_names = [n for n in named if ".policy." in n]
    trainable = [n for n, p in named.items() if p.requires_grad]
    snap_ref = {n: named[n].detach().float().cpu().clone() for n in ref_names}
    snap_pol = {n: named[n].detach().float().cpu().clone() for n in pol_names}
    full, rows = build_dataset(REPO / cfg["data"]["pairs"], cfg["seed"])
    pos = {r["pair_id"]: i for i, r in enumerate(rows)}
    S = pre["worst_case_sets"]
    order = S["longest_seq_8"] + S["longest_completion_8"] + S["longest_seq_8"]
    ds = full.select([pos[i] for i in order])
    out = RUNS / f"diag_feasibility_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "DO_NOT_USE.txt").write_text("DPO 可行性复测的临时权重（3 个优化步，最坏情况样本）。不得用于正式训练、初始化或任何评测。\n", encoding="utf-8")

    class Mem(TrainerCallback):
        def __init__(self):
            self.events = []

        def _rec(self, ev, state):
            f, _ = torch.cuda.mem_get_info()
            self.events.append({"event": ev, "step": state.global_step, "allocated_mib": round(torch.cuda.memory_allocated() / 2**20),
                                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                                "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20), "device_free_mib": round(f / 2**20)})

        def on_step_end(self, args, state, control, **kw):
            self._rec("step_end", state)

        def on_save(self, args, state, control, **kw):
            self._rec("save", state)

    dcfg = DPOConfig(output_dir=str(out), model_adapter_name="policy", ref_adapter_name="reference", beta=cfg["train"]["beta"], loss_type="sigmoid",
                     learning_rate=cfg["train"]["learning_rate"], lr_scheduler_type="cosine", warmup_ratio=0.1, num_train_epochs=1, max_steps=3,
                     per_device_train_batch_size=1, gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
                     max_length=cfg["data"]["max_length"], max_prompt_length=cfg["data"]["max_prompt_length"],
                     use_logits_to_keep=cfg["train"]["use_logits_to_keep"], gradient_checkpointing=True,
                     gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True, optim="paged_adamw_8bit", logging_steps=1,
                     save_strategy="steps", save_steps=1, save_only_model=True, report_to=[], seed=cfg["seed"], remove_unused_columns=False,
                     dataloader_num_workers=0)
    cb = Mem()
    trainer = DPOTrainer(model=model, ref_model=None, args=dcfg, train_dataset=ds, processing_class=tok, callbacks=[cb])
    # 实际 TRL 输入与预检一致
    _, tk, _ = tokenize_all(tok, cfg, cfg["data"]["max_prompt_length"])
    tk_by = {rows[i]["pair_id"]: tk[i] for i in range(len(rows))}
    same = sum(all(trainer.train_dataset[k][f] == tk_by[pid][f] for f in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"))
               for k, pid in enumerate(order))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ok, err = True, None
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        ok, err = False, f"CUDA OOM: {str(e)[:400]}"
    wall = round(time.time() - t0, 1)
    named = dict(model.named_parameters())
    ref_maxdiff = max(float((named[n].detach().float().cpu() - snap_ref[n]).abs().max()) for n in ref_names)
    pol_changed = sum(bool((named[n].detach().float().cpu() - snap_pol[n]).abs().max() > 0) for n in pol_names)
    saved = sorted(str(p.relative_to(out)) for p in out.rglob("adapter_model.safetensors"))
    init_t = load_file(str(init / "adapter_model.safetensors"))
    ref_saved_equal = []
    for s in saved:
        if "reference" in s:
            st = load_file(str(out / s))
            ref_saved_equal.append(all(torch.equal(st[k].float(), init_t[k].float()) for k in init_t if k in st) and len(st) == len(init_t))
    logs = trainer.state.log_history
    limit = round(total * frac / 2**20)
    peak_res = round(torch.cuda.max_memory_reserved() / 2**20)
    rec = {"when": time.strftime("%F %T"), "seed": seed, "ok": ok, "error": err, "purpose": "DPO 可行性复测（非正式训练；临时权重 DO_NOT_USE）",
           "init_adapter": str(init), "init_adapter_sha256": init_sha, "pairs_sha256": FROZEN_PAIRS_SHA,
           "data_order": {"step1": "longest_seq_8", "step2": "longest_completion_8", "step3": "longest_seq_8（重复，lr > 0 后看 reward 是否非零）"},
           "pair_ids": order, "trl_tokenization_matches_precheck": f"{same}/{len(order)}",
           "config": {"beta": dcfg.beta, "lr": dcfg.learning_rate, "max_length": dcfg.max_length, "max_prompt_length": dcfg.max_prompt_length,
                      "use_logits_to_keep": dcfg.use_logits_to_keep, "precompute_ref_log_probs": dcfg.precompute_ref_log_probs,
                      "optim": dcfg.optim, "ga": dcfg.gradient_accumulation_steps, "P6_MEM_FRACTION": frac,
                      "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")},
           "steps": trainer.state.global_step, "wall_time_s": wall,
           "gpu": {"total_mib": round(total / 2**20), "device_free_before_load_mib": round(free0 / 2**20),
                   "other_processes_mib_approx": round((total - free0) / 2**20), "process_limit_mib": limit, "load_peak_mib": load_peak,
                   "train_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20), "train_peak_reserved_mib": peak_res,
                   "headroom_reserved_vs_limit_mib": limit - peak_res},
           "memory_events": cb.events,
           "adapter_check": {"trainable_param_tensors": len(trainable), "trainable_all_policy": all(".policy." in n for n in trainable),
                             "reference_requires_grad_any": any(named[n].requires_grad for n in ref_names),
                             "reference_max_abs_change": ref_maxdiff, "policy_tensors_changed": f"{pol_changed}/{len(pol_names)}",
                             "saved_reference_equals_init_adapter": ref_saved_equal},
           "checkpoint_saved": saved, "has_preference_metrics": any("rewards/margins" in x for x in logs),
           "logs": [x for x in logs if "loss" in x],
           "versions": {"trl": trl.__version__, "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__}}
    (OUT / f"memory_s{seed}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN", json.dumps({k: rec[k] for k in ("seed", "ok", "error", "steps", "wall_time_s", "gpu", "adapter_check", "trl_tokenization_matches_precheck")},
                            ensure_ascii=False), flush=True)
    return 0 if ok else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["precheck", "memory"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return precheck() if a.mode == "precheck" else memory(a.seed)


if __name__ == "__main__":
    raise SystemExit(main())
