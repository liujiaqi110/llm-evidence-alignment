"""DPO 最坏情况显存测试（GPU；只跑 1 个优化步 = 8 对，测试训练前向、反向、优化器步与检查点保存；不是正式训练）。

    P6_MEM_FRACTION=0.74 python -m rag_finetuning.experiments.p6_v5.diagnostics.dpo_worst_case_memory --set longest_seq_8 --max-length 1024

--set：truncation_audit 产出的 worst_case_sets.json 中的 longest_seq_8（[prompt+回答] 最长 8 对）或 longest_completion_8（回答 token 最长 8 对）。
配置与正式 DPO 相同（修订 3 提议：use_logits_to_keep=True）；save_steps=1，测试保存检查点；输出写到
workspace/runs/p6_v5/diag_worst_<set>_<len>/（只放诊断用适配器副本，不用于任何评测）。
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from rag_finetuning.experiments.p6_v5.diagnostics.dpo_diag_common import PAIRS, V5, dpo_config, load_policy_ref, tokenizer
from rag_finetuning.experiments.p6_v5.scripts.train_dpo_v5 import build_dataset

OUT = V5 / "diagnostics" / "dpo_worst_case"
RUNS = Path("workspace/runs/p6_v5")


class Mem(TrainerCallback):
    def __init__(self):
        self.events = []

    def _rec(self, ev, state):
        self.events.append({"event": ev, "step": state.global_step, "allocated_mib": round(torch.cuda.memory_allocated() / 2**20),
                            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)})

    def on_step_end(self, args, state, control, **kw):
        self._rec("step_end", state)

    def on_save(self, args, state, control, **kw):
        self._rec("save", state)


def main(argv=None) -> int:
    from trl import DPOTrainer
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["longest_seq_8", "longest_completion_8", "longest_untruncated_8", "longest_completion_untruncated_8"])
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--precompute-ref", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    sets = json.loads((V5 / "diagnostics/truncation/worst_case_sets.json").read_text(encoding="utf-8"))
    want = sets[a.set]
    tok = tokenizer()
    full, rows = build_dataset(PAIRS, 42)
    idx = [i for i, r in enumerate(rows) if r["pair_id"] in want]
    assert len(idx) == 8, idx
    ds = full.select(idx)
    torch.cuda.reset_peak_memory_stats()
    model = load_policy_ref()
    load_peak = round(torch.cuda.max_memory_allocated() / 2**20)
    out = RUNS / f"diag_worst_{a.set}_{a.max_length}{a.tag}"
    cb = Mem()
    trainer = DPOTrainer(model=model, ref_model=None, train_dataset=ds, processing_class=tok, callbacks=[cb],
                         args=dpo_config(out, max_length=a.max_length, max_prompt_length=min(896, a.max_length - 128),
                                         max_steps=1, save_strategy="steps", save_steps=1, precompute_ref_log_probs=a.precompute_ref))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ok, err = True, None
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        ok, err = False, f"CUDA OOM: {str(e)[:300]}"
    saved = sorted(str(p.relative_to(out)) for p in out.rglob("adapter_model.safetensors")) if out.exists() else []
    logs = trainer.state.log_history
    rec = {"set": a.set, "tag": a.tag, "precompute_ref_log_probs": a.precompute_ref, "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""), "pair_ids": want, "max_length": a.max_length, "use_logits_to_keep": True, "ok": ok, "error": err,
           "steps": trainer.state.global_step, "wall_time_s": round(time.time() - t0, 1), "load_peak_mib": load_peak,
           "train_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
           "train_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
           "limit_mib": round(torch.cuda.get_device_properties(0).total_memory * 0.74 / 2**20),
           "memory_events": cb.events, "checkpoint_saved": saved, "has_preference_metrics": any("rewards/margins" in x for x in logs),
           "logs": [x for x in logs if "loss" in x]}
    (OUT / f"{a.set}_{a.max_length}{a.tag}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN", json.dumps({k: rec[k] for k in ("set", "max_length", "tag", "ok", "error", "train_peak_allocated_mib", "train_peak_reserved_mib", "checkpoint_saved")},
                            ensure_ascii=False), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
