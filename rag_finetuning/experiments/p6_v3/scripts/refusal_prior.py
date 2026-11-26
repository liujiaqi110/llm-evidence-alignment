"""拒答先验诊断（与 P6-v1 审查相同的方法）：在 test_v3 的“应回答”题上，比较 Base 与 Adapter 对参考答案和固定拒答句的平均 token 对数概率。

    python -m rag_finetuning.experiments.p6_v3.scripts.refusal_prior --size 1.5b --adapter <dir> --out <json>

说明：这是对训练机制的诊断，不是质量指标；只读取 test_v3 的问题与参考答案（模型输出已经生成并冻结，诊断不影响任何选择）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from rag_finetuning.common import CANONICAL_REFUSAL, messages_for, read_jsonl
from rag_finetuning.experiments.p6_v3.scripts.run_eval import M15, M3

V3 = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["1.5b", "3b"], required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    mid, rev = M15 if a.size == "1.5b" else M3
    tok = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(mid, revision=rev, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16,
                                                local_files_only=True)
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    rows = [r for r in read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl") if r["behavior"] == "answer"]

    def lp(msgs, target):
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        t = tok(target + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p + t], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0, len(p) - 1:-1].float()
        return torch.log_softmax(logits, -1).gather(1, torch.tensor(t, device="cuda")[:, None]).mean().item()

    res = []
    for r in rows:
        m = messages_for(r)
        row = {"id": r["id"]}
        with model.disable_adapter():
            row["base_ref"], row["base_refusal"] = lp(m, r["reference_answer"]), lp(m, CANONICAL_REFUSAL)
        row["ft_ref"], row["ft_refusal"] = lp(m, r["reference_answer"]), lp(m, CANONICAL_REFUSAL)
        res.append(row)
    n = len(res)
    mean = lambda k: round(sum(x[k] for x in res) / n, 3)  # noqa: E731
    out = {"size": a.size, "adapter": a.adapter, "n_answerable": n,
           "mean_logprob_reference": {"base": mean("base_ref"), "ft": mean("ft_ref")},
           "mean_logprob_refusal": {"base": mean("base_refusal"), "ft": mean("ft_refusal")},
           "items_refusal_more_likely_than_reference": {"base": sum(x["base_refusal"] > x["base_ref"] for x in res),
                                                        "ft": sum(x["ft_refusal"] > x["ft_ref"] for x in res)},
           "per_item": res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
