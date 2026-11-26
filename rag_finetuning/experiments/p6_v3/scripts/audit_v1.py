"""P6-v1 训练管线与数据审查：每项结论都输出可复核的证据（不凭经验下结论）。

    python -m rag_finetuning.experiments.p6_v3.scripts.audit_v1 [--gpu]

CPU 检查（只需分词器）：
  1 Chat Template：tokenize=True 与“渲染文本再分词”两条路径是否逐 token 一致；system 是否唯一
  2 Loss 掩码：只有 assistant 回答 + <|im_end|> 参与 loss
  3 EOS / 特殊 Token：<|im_end|> 是否为单个特殊 token、生成配置的 eos、是否有重复特殊 token
  4 Padding：padding token 的 label 与 attention 是否屏蔽
  5 打包：是否存在多样本拼接（样本串扰）
  6 LoRA target_modules：是否覆盖 Qwen2 全部线性层（层数 × 7）
  7 验证集来源泄漏；8 训练目标重复度；9 拒答样本的剩余证据是否其实可回答；10 过拟合迹象
GPU 检查（--gpu）：
  11 在 P6-v1 测试集 / P6-v2 盲测集的“应回答”题上，比较 Base 与 v1 Adapter 对
     参考答案与固定拒答句的平均 token 对数概率（直接测量“拒答先验”是否被训练抬高）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rag_finetuning.common import BUILT, CANONICAL_REFUSAL, REPO, messages_for, read_jsonl
from rag_finetuning.evaluation.generate import MODEL_ID, REVISION
from rag_finetuning.scripts.train_qlora import Collator, encode

OUT = Path(__file__).resolve().parents[1] / "results" / "v1_audit"
RUNS = Path("workspace/runs")


def cpu_checks(tok) -> dict:
    train = read_jsonl(BUILT / "train.jsonl")
    val = read_jsonl(BUILT / "val.jsonl")
    ev = {}
    # 1 chat template
    r = train[0]
    msgs = messages_for(r)
    a = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    b = tok(text, add_special_tokens=False)["input_ids"]
    ev["1_chat_template"] = {
        "tokenize_paths_identical": a == b, "n_prompt_tokens": len(a),
        "system_blocks": text.count("<|im_start|>system"), "default_system_injected": "You are Qwen" in text,
        "prompt_tail_repr": repr(text[-40:]), "prompt_head_repr": repr(text[:60])}
    # 2 / 3 masking & EOS
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    enc = [encode(tok, x, 1024) for x in train]
    bad_mask = []
    for x, e in zip(train, enc):
        lab = [t for t in e["labels"] if t != -100]
        n_prompt = sum(1 for t in e["labels"] if t == -100)
        ok = (n_prompt == e["n_prompt"] and lab == e["input_ids"][e["n_prompt"]:] and lab[-1] == im_end
              and tok.decode(lab[:-1]) == x["reference_answer"])
        if not ok:
            bad_mask.append(x["id"])
    specials = Counter()
    for e in enc:
        specials[(e["input_ids"].count(tok.convert_tokens_to_ids("<|im_start|>")), e["input_ids"].count(im_end),
                  e["input_ids"].count(eot))] += 1
    gen_cfg = json.loads((Path(tok.name_or_path) / "generation_config.json").read_text()) if Path(tok.name_or_path).exists() else None
    ev["2_loss_mask"] = {"samples": len(enc), "label_tokens_equal_answer_plus_im_end": len(enc) - len(bad_mask),
                         "violations": bad_mask,
                         "label_token_share_of_sequence": round(sum(x["n_completion"] for x in enc) / sum(len(x["input_ids"]) for x in enc), 4)}
    ev["3_eos_special_tokens"] = {
        "im_end_single_token": tok("<|im_end|>", add_special_tokens=False)["input_ids"] == [im_end],
        "im_end_id": im_end, "endoftext_id": eot, "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
        "eos_token": tok.eos_token, "generation_config_eos": (gen_cfg or {}).get("eos_token_id"),
        "special_token_counts_(im_start,im_end,endoftext)->n_samples": {str(k): v for k, v in specials.items()},
        "note": "每条样本应为 im_start×3（system/user/assistant）、im_end×3（含回答结尾）、endoftext×0"}
    # 4 padding
    col = Collator(tok.pad_token_id)
    short, long_ = min(enc, key=lambda e: len(e["input_ids"])), max(enc, key=lambda e: len(e["input_ids"]))
    batch = col([{k: short[k] for k in ("input_ids", "attention_mask", "labels")},
                 {k: long_[k] for k in ("input_ids", "attention_mask", "labels")}])
    L = len(short["input_ids"])
    pad_lab = batch["labels"][0, L:].tolist()
    pad_att = batch["attention_mask"][0, L:].tolist()
    ev["4_padding"] = {"padded_len": batch["input_ids"].shape[1], "short_len": L, "long_len": len(long_["input_ids"]),
                       "pad_labels_all_-100": all(t == -100 for t in pad_lab), "pad_attention_all_0": all(t == 0 for t in pad_att),
                       "padding_side": "right", "note": "因果注意力 + 右侧 padding + attention_mask=0：真实 token 看不到 padding"}
    # 5 packing
    ev["5_packing"] = {"packing_used": False, "evidence": "Collator 对每条样本单独 padding；ListDS 一条样本 = 一条序列；"
                       "所有样本 im_start 计数恒为 3（见 3），不存在两条样本拼在一条序列中"}
    # 7 leakage
    docs = {s: {d for x in read_jsonl(BUILT / f"{s}.jsonl") for d in x["source_docs"]} for s in ("train", "val", "test")}
    ev["7_val_leakage"] = {"train∩val_docs": sorted(docs["train"] & docs["val"]), "train∩test_docs": sorted(docs["train"] & docs["test"]),
                           "val∩test_docs": sorted(docs["val"] & docs["test"])}
    # 8 repetition
    targets = Counter(x["reference_answer"] for x in train)
    top = targets.most_common(1)[0]
    refuse_tokens = sum(e["n_completion"] for x, e in zip(train, enc) if x["behavior"] == "refuse")
    q_counts = Counter(x["question"] for x in train)
    beh = Counter(x["behavior"] for x in train)
    ev["8_repetition"] = {
        "n_train": len(train), "behavior_counts": dict(beh),
        "most_common_target_count": top[1], "most_common_target_is_canonical_refusal": top[0] == CANONICAL_REFUSAL,
        "distinct_targets": len(targets), "refusal_share_of_samples": round(beh["refuse"] / len(train), 3),
        "refusal_share_of_loss_tokens": round(refuse_tokens / sum(e["n_completion"] for e in enc), 3),
        "questions_appearing_twice_with_different_behavior": sum(1 for q, n in q_counts.items() if n > 1),
        "answer_target_tokens_mean": round(sum(e["n_completion"] for x, e in zip(train, enc) if x["behavior"] == "answer")
                                           / beh["answer"], 1),
        "val_behavior_counts": dict(Counter(x["behavior"] for x in val))}
    # 9 refusal labels: is the answer actually still in the remaining evidence?
    by_id = {x["id"]: x for x in train}
    suspicious = []
    for x in train:
        if x["behavior"] != "refuse":
            continue
        evtext = re.sub(r"\s+", "", " ".join(e["text"] for e in x["evidence"]))
        src = by_id.get(x.get("derived_from") or "")
        facts = [f for f in (src["facts"] if src else []) for f in [f.split("|")[0]]]
        hit = [f for f in facts if re.sub(r"\s+", "", f) in evtext]
        if hit:
            suspicious.append({"id": x["id"], "derived_from": x.get("derived_from"), "facts_still_in_evidence": hit})
    ev["9_refusal_label_check"] = {
        "refuse_samples": beh["refuse"], "derived_gold_removed": sum(1 for x in train if x["generation_method"] == "derived_gold_removed"),
        "derived_with_original_facts_still_in_evidence": suspicious,
        "note": "必要事实（数字 / 关键词）仍出现在剩余证据中，说明“删除金标准 = 不可回答”的标签可能错误；需逐条人工确认"}
    return ev


def train_dynamics() -> dict:
    out = {}
    for run in ("qlora_r8_v1", "qlora_r8_v2_noderived"):
        rec = json.loads((RUNS / run / "run_record.json").read_text(encoding="utf-8"))
        log = read_jsonl(RUNS / run / "train_log.jsonl")
        steps = [x for x in log if "loss" in x]
        per_epoch = {}
        for x in steps:
            per_epoch.setdefault(int(min(x["epoch"] - 1e-9, 2.999)), []).append(x["loss"])
        out[run] = {"train_loss_by_epoch_mean": {k + 1: round(sum(v) / len(v), 3) for k, v in sorted(per_epoch.items())},
                    "eval_loss_curve": [round(x["eval_loss"], 4) for x in log if "eval_loss" in x],
                    "lora_target_modules": rec["config"]["lora"]["target_modules"], "trainable_params": rec["trainable_params"]}
    return out


def gpu_logprob(tok) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, quantization_config=q, device_map={"": 0},
                                                dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(RUNS / "qlora_r8_v1" / "adapter")).eval()
    sets = {"p6v1_test": read_jsonl(BUILT / "test.jsonl"),
            "p6v2_blind": read_jsonl(REPO / "rag_finetuning/experiments/p6_v2/blind/built/blind.jsonl")}

    def mean_lp(msgs, target):
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        t = tok(target + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + t], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0, len(p) - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(t, device="cuda")[:, None]).squeeze(1)
        return lp.mean().item()

    out = {}
    for name, rows in sets.items():
        res = []
        for r in rows:
            if r["behavior"] != "answer":
                continue
            msgs = messages_for(r)
            row = {"id": r["id"]}
            for tag, ctx in (("base", model.disable_adapter()), ("v1", None)):
                if ctx is not None:
                    with ctx:
                        row[f"{tag}_ref"], row[f"{tag}_refusal"] = mean_lp(msgs, r["reference_answer"]), mean_lp(msgs, CANONICAL_REFUSAL)
                else:
                    row[f"{tag}_ref"], row[f"{tag}_refusal"] = mean_lp(msgs, r["reference_answer"]), mean_lp(msgs, CANONICAL_REFUSAL)
            res.append(row)
        n = len(res)
        out[name] = {
            "n_answerable": n,
            "mean_logprob_reference": {"base": round(sum(x["base_ref"] for x in res) / n, 3), "v1": round(sum(x["v1_ref"] for x in res) / n, 3)},
            "mean_logprob_refusal": {"base": round(sum(x["base_refusal"] for x in res) / n, 3), "v1": round(sum(x["v1_refusal"] for x in res) / n, 3)},
            "items_where_refusal_more_likely_than_reference": {"base": sum(x["base_refusal"] > x["base_ref"] for x in res),
                                                               "v1": sum(x["v1_refusal"] > x["v1_ref"] for x in res)},
            "per_item": res}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cpu": cpu_checks(tok), "train_dynamics": train_dynamics()}
    if a.gpu:
        res["gpu_logprob"] = gpu_logprob(tok)
    (OUT / "audit_v1.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    slim = json.loads(json.dumps(res))
    for v in (slim.get("gpu_logprob") or {}).values():
        v.pop("per_item", None)
    print(json.dumps(slim, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
