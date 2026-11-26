"""P6-v4 QLoRA 训练：复用 P6-v3 已验证的 train_v3（同一管线、同一配方），只替换停止点选择规则为 v4 预注册规则：
开发集 dev_v4（103 条）上评分器 v3 的 content_ok 最高 → multi_content_complete 更高 → 应回答类整体拒答更少 → 更早的 epoch。

    python -m rag_finetuning.experiments.p6_v4.scripts.train_v4 --config rag_finetuning/experiments/p6_v4/configs/v4_3b_s42.yaml
    python -m rag_finetuning.experiments.p6_v4.scripts.train_v4 --dev-only --group A|B --out <json>   # 开发集基线（不训练）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from rag_finetuning.common import REPO, messages_for, read_jsonl
from rag_finetuning.experiments.p6_v3.scripts import train_v3
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import score_item_v3

V4 = Path(__file__).resolve().parents[1]
DEV = V4 / "data" / "v4" / "dev.jsonl"
M3 = ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1")
GEN = {"max_new_tokens": 384, "do_sample": False, "repetition_penalty": 1.0}


def dev_scores(model, tok, dev):
    agg = {"dev_content_ok": 0, "dev_multi_content": 0, "dev_multi_n": 0, "dev_whole_refusal": 0, "dev_refusal_ok": 0}
    per = []
    for r in dev:
        ids = tok.apply_chat_template(messages_for(r), add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, **GEN)
        ans = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()
        s = score_item_v3(r, ans)
        agg["dev_content_ok"] += int(s["content_ok"])
        agg["dev_multi_n"] += int(s["is_multi"])
        agg["dev_multi_content"] += int(bool(s["multi_content_complete"]))
        agg["dev_whole_refusal"] += int(bool(s["whole_refusal_on_answerable"]))
        agg["dev_refusal_ok"] += int(bool(s["refusal_ok"]))
        per.append({"id": r["id"], "content_ok": s["content_ok"], "multi": s["multi_content_complete"]})
    agg["dev_n"] = len(dev)
    return agg, per


def load(adapter=None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(M3[0], revision=M3[1], local_files_only=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(M3[0], revision=M3[1], quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16,
                                             local_files_only=True)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter, adapter_name="a0")
    return tok, m.eval()


def select_epoch_v4(cfg, out, _tok_dir):
    dev = read_jsonl(REPO / cfg["data"]["dev"])
    ckpts = sorted(out.glob("checkpoint_epoch*"), key=lambda p: int(p.name.replace("checkpoint_epoch", "")))
    tok, base = load(None)
    from peft import PeftModel
    model, res = None, []
    for ck in ckpts:
        if model is None:
            model = PeftModel.from_pretrained(base, str(ck), adapter_name=ck.name)
        else:
            model.load_adapter(str(ck), adapter_name=ck.name)
        model.set_adapter(ck.name)
        model.eval()
        agg, _ = dev_scores(model, tok, dev)
        agg["epoch"] = int(ck.name.replace("checkpoint_epoch", ""))
        res.append(agg)
        print("DEV", agg, flush=True)
    best = sorted(res, key=lambda x: (-x["dev_content_ok"], -x["dev_multi_content"], x["dev_whole_refusal"], x["epoch"]))[0]
    return {"rule": "max dev content_ok(v3) → max multi_content_complete → min whole_refusal → earliest", "per_epoch": res,
            "selected_epoch": best["epoch"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--group", choices=["A", "B"])
    ap.add_argument("--out")
    a, rest = ap.parse_known_args(argv)
    if a.dev_only:
        adapter = None if a.group == "A" else "workspace/runs/p6_v3/v3_3b_full/adapter"
        tok, model = load(adapter)
        agg, per = dev_scores(model, tok, read_jsonl(DEV))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"group": a.group, "adapter": adapter, **agg, "per_item": per}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print("DEVBASE", a.group, agg)
        return 0
    train_v3.select_epoch = select_epoch_v4          # 只替换选择规则，其余训练代码与 v3 完全相同
    frac = os.environ.get("P6_MEM_FRACTION")
    if frac:
        # 只限制 CUDA 缓存分配器的上限（超限时先释放缓存再重试），避免 WDDM 显存溢出到共享内存；不改变任何超参数与计算
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)
        print(f"MEM_FRACTION {frac} -> {float(frac) * torch.cuda.get_device_properties(0).total_memory / 2**20:.0f} MiB", flush=True)
    return train_v3.main(["--config", a.config] + rest)


if __name__ == "__main__":
    sys.exit(main())
