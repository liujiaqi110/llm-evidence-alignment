"""P6-v5 实验一：针对性 SFT。复用 P6-v3 train_v3（同一管线、同一配方），只替换停止点选择规则为 v5 预注册规则：
hard_dev 上评分器 v4 的 multi_complete_v5 最高 → content_ok_v4 最高 → 应回答类整体拒答最少 → 更早的 epoch。

    P6_MEM_FRACTION=0.74 python -m rag_finetuning.experiments.p6_v5.scripts.train_sft_v5 --config rag_finetuning/experiments/p6_v5/configs/sft_3b_s42.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from rag_finetuning.common import REPO, messages_for, read_jsonl
from rag_finetuning.experiments.p6_v3.scripts import train_v3
from rag_finetuning.experiments.p6_v4.scripts.train_v4 import GEN, load
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4


def dev_scores_v5(model, tok, dev, save_path=None):
    """save_path：逐题保存回答与关键分数（2026-11-24 加入；只记录，不影响选择规则）。"""
    import json
    rows = []
    agg = {"dev_multi_v5": 0, "dev_multi_n": 0, "dev_content_v4": 0, "dev_whole_refusal": 0, "dev_refusal_ok": 0, "dev_conflict_ok": 0}
    for r in dev:
        ids = tok.apply_chat_template(messages_for(r), add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, **GEN)
        ans = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()
        s = score_item_v4(r, ans)
        rows.append({"id": r["id"], "answer": ans, "output_tokens": int(o.shape[1] - ids.shape[1]), "scorer": s["scorer_v4"],
                     "multi_complete_v5": s["multi_complete_v5"], "content_ok_v4": s["content_ok_v4"], "missing_facts": s["missing_facts"]})
        agg["dev_multi_n"] += int(s["is_multi_v5"])
        agg["dev_multi_v5"] += int(bool(s["multi_complete_v5"]))
        agg["dev_content_v4"] += int(s["content_ok_v4"])
        agg["dev_whole_refusal"] += int(bool(s["whole_refusal_on_answerable"]))
        agg["dev_refusal_ok"] += int(bool(s["refusal_ok"]))
        agg["dev_conflict_ok"] += int(bool(s["conflict_true_ok"]))
    agg["dev_n"] = len(dev)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return agg


def select_epoch_v5(cfg, out, _tok_dir):
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
        agg = dev_scores_v5(model, tok, dev, save_path=out / "dev_eval" / f"{ck.name}.jsonl")
        agg["epoch"] = int(ck.name.replace("checkpoint_epoch", ""))
        res.append(agg)
        print("DEV", agg, flush=True)
    best = sorted(res, key=lambda x: (-x["dev_multi_v5"], -x["dev_content_v4"], x["dev_whole_refusal"], x["epoch"]))[0]
    return {"rule": "max hard_dev multi_complete_v5 → max content_ok_v4 → min whole_refusal → earliest", "per_epoch": res,
            "selected_epoch": best["epoch"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a, rest = ap.parse_known_args(argv)
    train_v3.select_epoch = select_epoch_v5          # 只替换选择规则，其余训练代码与 v3 / v4 完全相同
    frac = os.environ.get("P6_MEM_FRACTION")
    if frac:
        torch.cuda.set_per_process_memory_fraction(float(frac), 0)
        print(f"MEM_FRACTION {frac}", flush=True)
    return train_v3.main(["--config", a.config] + rest)


if __name__ == "__main__":
    sys.exit(main())
