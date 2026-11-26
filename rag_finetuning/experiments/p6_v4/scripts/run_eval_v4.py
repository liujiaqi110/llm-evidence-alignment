"""P6-v4 评估：冻结输入 / 统一推理（含“工程方案二”提示变体）/ 评分器 v3 分析（含“工程方案一”后处理变体）。

    python -m rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 freeze --name test_v4 --src <built.jsonl>
    python -m rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 gen --group C42 --input test_v4 --out <dir>/test_v4/C42
    python -m rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 analyze --input test_v4 --ref <built.jsonl> --runs <dir>/test_v4 --out <out>

组：A 3B Base | B 3B + v3 Adapter | C42 / C43 3B + v4 Adapter（种子 42 / 43）| D Ollama qwen2.5:7b（参考）
    A_E2 / C42_E2：同 A / C42，但在用户消息末尾追加冲突预检测提示（eng.conflict_hint，规则生成，不看答案）
分析时对每组再计算 “+E1” 变体：用 eng.coverage_check 对已有输出做后处理（不重新生成）。
所有组：平台 rag_v1.0 Prompt（E2 变体仅追加提示段落）、相同证据与顺序、贪心、max_new_tokens 384。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl, write_jsonl
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as sv2
from rag_finetuning.experiments.p6_v4.scripts import eng
from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import SCORER_VERSION, score_item_v3
from rag_finetuning.compat.rag_pipeline import build_messages

V4 = Path(__file__).resolve().parents[1]
FROZEN = V4 / "data" / "frozen"
RUNS = "workspace/runs"
M3 = ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1")
M15 = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
GROUPS = {
    "A": {"desc": "3B 原始模型", "model": M3, "adapter": None},
    "B": {"desc": "3B + v3 QLoRA（P6-v3 E 组）", "model": M3, "adapter": f"{RUNS}/p6_v3/v3_3b_full/adapter"},
    "C42": {"desc": "3B + v4 QLoRA（种子 42）", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s42/adapter"},
    "C43": {"desc": "3B + v4 QLoRA（种子 43）", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s43/adapter"},
    "D": {"desc": "Ollama qwen2.5:7b（Q4_K_M）参考组", "ollama": "qwen2.5:7b"},
    "A_E2": {"desc": "3B 原始模型 + 工程方案二（冲突预检测提示）", "model": M3, "adapter": None, "hint": True},
    "C42_E2": {"desc": "3B + v4（种子 42）+ 工程方案二", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s42/adapter", "hint": True},
    "B_E2": {"desc": "3B + v3 + 工程方案二", "model": M3, "adapter": f"{RUNS}/p6_v3/v3_3b_full/adapter", "hint": True},
}
GEN = {"max_new_tokens": 384, "do_sample": False, "repetition_penalty": 1.0}
OLLAMA_OPTS = {"temperature": 0, "seed": 42, "num_predict": 384, "num_ctx": 8192}


def item_hash(r):
    return hashlib.sha256(json.dumps({"question": r["question"], "evidence": [[e["cid"], e["source"], e["text"]] for e in r["evidence"]]},
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def msgs(r, hint=False):
    m = build_messages(r["question"], [(e["cid"], e["text"]) for e in r["evidence"]], {e["cid"]: e["source"] for e in r["evidence"]})
    if hint:
        m[1]["content"] += eng.conflict_hint(r["evidence"])
    return m


def cmd_freeze(a):
    rows = read_jsonl(Path(a.src))
    refs = [r["reference_answer"].split("\n\n引用：")[0] for r in rows if r["behavior"] != "refuse"]
    for r in rows:
        for h in (False, True):
            blob = "\n".join(m["content"] for m in msgs(r, h))
            assert not any(len(x) > 12 and x in blob for x in refs), f"{r['id']}: 参考答案出现在 Prompt 中"
    frozen = [{"id": r["id"], "question": r["question"], "evidence": [{k: e[k] for k in ("cid", "source", "doc", "text")} for e in r["evidence"]],
               "input_sha256": item_hash(r)} for r in rows]
    FROZEN.mkdir(parents=True, exist_ok=True)
    p = FROZEN / f"{a.name}.jsonl"
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in frozen)
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} 已存在且内容不同：拒绝覆盖")
    p.write_text(text, encoding="utf-8", newline="\n")
    man = {"name": a.name, "source": a.src, "source_sha256": hashlib.sha256(Path(a.src).read_bytes()).hexdigest(),
           "frozen_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n": len(frozen), "contains_reference_answers": False,
           "items": {x["id"]: x["input_sha256"] for x in frozen}}
    (FROZEN / f"{a.name}.manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("name", "n", "frozen_sha256")}, ensure_ascii=False))


def cmd_gen(a):
    g = GROUPS[a.group]
    p = FROZEN / f"{a.input}.jsonl"
    man = json.loads((FROZEN / f"{a.input}.manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(p.read_bytes()).hexdigest() != man["frozen_sha256"]:
        raise SystemExit("冻结输入哈希不一致")
    out = Path(a.out)
    if (out / "meta.json").exists():
        raise SystemExit(f"{out} 已有结果：每组只允许生成一次")
    rows = read_jsonl(p)
    from transformers import AutoTokenizer
    tok15 = AutoTokenizer.from_pretrained(M15[0], revision=M15[1], local_files_only=True)
    if "ollama" in g:
        from rag_finetuning.compat.local_llm import OllamaClient
        c = OllamaClient(model=g["ollama"], timeout_s=900)
        info = c.model_info()

        def run(m):
            r = c.chat(m, dict(OLLAMA_OPTS))
            return r.text, {"latency_ms": r.total_ms, "output_tokens": r.output_tokens}
        settings = {"backend": "ollama", **OLLAMA_OPTS, "model_info": info}
        name = f"ollama:{g['ollama']}"
    else:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        mid, rev = g["model"]
        tok = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16,
                                                     local_files_only=True)
        if g["adapter"]:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, g["adapter"])
        model.eval()
        load_peak = round(torch.cuda.max_memory_allocated() / 2**20)

        def run(m):
            ids = tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                o = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, **GEN)
            torch.cuda.synchronize()
            new = o[0, ids.shape[1]:]
            return tok.decode(new, skip_special_tokens=True).strip(), {
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "output_tokens": int(new.shape[0]),
                "gpu_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
        settings = {"backend": "transformers", **GEN, "model_id": mid, "revision": rev, "adapter": g["adapter"],
                    "quantization": "bitsandbytes 4-bit NF4 + double quant, bf16 compute", "load_peak_alloc_mib": load_peak}
        name = mid.split("/")[-1] + (f"+{Path(g['adapter']).parent.name}" if g["adapter"] else "")
    out.mkdir(parents=True, exist_ok=True)
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        m = msgs(r, g.get("hint", False))
        ans, extra = run(m)
        res.append({"id": r["id"], "group": a.group, "model": name, "answer": ans, "input_sha256": r["input_sha256"],
                    "hint": g.get("hint", False), "input_tokens_qwen_tokenizer": len(tok15.apply_chat_template(m, add_generation_prompt=True, tokenize=True)),
                    **extra})
        print(f"[{i}/{len(rows)}] {r['id']} {extra['latency_ms']:.0f}ms", flush=True)
    write_jsonl(out / "generations.jsonl", res)
    meta = {"group": a.group, "desc": g["desc"], "model": name, "input": a.input, "frozen_sha256": man["frozen_sha256"],
            "generation_settings": settings, "hint_variant": g.get("hint", False), "n": len(res), "wall_s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if "ollama" in g:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", method="POST",
                                                      data=json.dumps({"model": g["ollama"], "keep_alive": 0}).encode()), timeout=60).read()
    print("DONE", json.dumps({k: meta[k] for k in ("group", "model", "n", "wall_s")}, ensure_ascii=False))


ANS = ("answer", "partial", "conflict")
METRICS = {  # 名称: (适用题, 取值键, 越低越好)
    "content_ok": (lambda r: True, "content_ok", False),
    "fact_complete(应回答类)": (lambda r: r["behavior"] in ANS, "complete", False),
    "multi_content_complete": (lambda r: True, "multi_content_complete", False),
    "multi_chunk_coverage": (lambda r: True, "multi_chunk_coverage", False),
    "fact_attribution_ok": (lambda r: r["behavior"] in ANS, "fact_attribution_ok", False),
    "false_refusal(answer 题)": (lambda r: r["behavior"] == "answer", "false_refusal", True),
    "whole_refusal(应回答类)": (lambda r: r["behavior"] in ANS, "whole_refusal_on_answerable", True),
    "refusal_ok": (lambda r: r["behavior"] == "refuse", "refusal_ok", False),
    "refusal_strict": (lambda r: r["behavior"] == "refuse", "refusal_strict", False),
    "partial_ok": (lambda r: r["behavior"] == "partial", "partial_ok", False),
    "claim_support_full(全分母)": (lambda r: r["behavior"] in ANS, "claim_support_full", False),
    "claim_support_and_complete": (lambda r: r["behavior"] in ANS, "claim_support_and_complete", False),
    "citation_valid": (lambda r: True, "citation_valid", False),
    "format_ok": (lambda r: True, "format_ok", False),
    "conflict_true_ok": (lambda r: r["behavior"] == "conflict", "conflict_true_ok", False),
    "version_override_ok": (lambda r: r.get("conflict_kind") == "version_override", "no_conflict_ok", False),
    "old_rule_still_applies_ok": (lambda r: r.get("conflict_kind") == "old_rule_still_applies", "no_conflict_ok", False),
    "scope_difference_ok": (lambda r: r.get("conflict_kind") == "scope_difference", "no_conflict_ok", False),
    "condition_difference_ok": (lambda r: r.get("conflict_kind") == "condition_difference", "no_conflict_ok", False),
    "same_conclusion_ok": (lambda r: r.get("conflict_kind") == "same_conclusion_despite_values", "no_conflict_ok", False),
    "general_exception_ok": (lambda r: r.get("subtype") == "general_exception", "content_ok", False),
    "injection_resisted": (lambda r: r.get("subtype") == "prompt_injection", "content_ok", False),
}


def cmd_analyze(a):
    man = json.loads((FROZEN / f"{a.input}.manifest.json").read_text(encoding="utf-8"))
    recs = {r["id"]: r for r in read_jsonl(Path(a.ref))}
    order = list(man["items"])
    runs = Path(a.runs)
    groups = [g for g in GROUPS if (runs / g / "generations.jsonl").exists()]
    data = {}
    for g in groups:
        gens = {x["id"]: x for x in read_jsonl(runs / g / "generations.jsonl")}
        meta = json.loads((runs / g / "meta.json").read_text(encoding="utf-8"))
        assert meta["frozen_sha256"] == man["frozen_sha256"], g
        assert all(gens[q]["input_sha256"] == man["items"][q] for q in order), g
        data[g] = {"meta": meta, "gens": gens, "rows": [{"rec": recs[q], "s": score_item_v3(recs[q], gens[q]["answer"]), "g": gens[q]} for q in order]}
        e1 = {q: eng.coverage_check(recs[q]["question"], recs[q]["evidence"], gens[q]["answer"]) for q in order}
        data[f"{g}+E1"] = {"meta": {**meta, "desc": meta["desc"] + " + 工程方案一（事实覆盖后处理）"}, "gens": gens,
                           "rows": [{"rec": recs[q], "s": score_item_v3(recs[q], e1[q]["answer"]), "g": {**gens[q], "answer": e1[q]["answer"]},
                                     "e1_added": e1[q]["added"]} for q in order]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"input": a.input, "n": len(order), "scorer": f"v{SCORER_VERSION}", "label": "自动规则评分，provisional（未经人工审核）",
            "groups": {}, "pairs": {}}
    for g, D in data.items():
        mets = {k: sv2.rate(x["s"][key] for x in D["rows"] if sel(x["rec"])) for k, (sel, key, low) in METRICS.items()}
        mets["claims_supported(按陈述)"] = {"k": sum(x["s"]["n_supported"] for x in D["rows"] if x["rec"]["behavior"] in ANS),
                                          "n": sum(x["s"]["n_claims"] for x in D["rows"] if x["rec"]["behavior"] in ANS)}
        mets["v2_content_ok"] = sv2.rate(x["s"]["v2"]["content_ok"] for x in D["rows"])
        lat = sorted(x["g"].get("latency_ms") or 0 for x in D["rows"])
        mets["cost"] = {"latency_ms_p50": statistics.median(lat), "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))],
                        "output_tokens_mean": round(statistics.mean((x["g"].get("output_tokens") or 0) for x in D["rows"]), 1),
                        "input_tokens_mean": round(statistics.mean(x["g"].get("input_tokens_qwen_tokenizer") or 0 for x in D["rows"]), 1),
                        "e1_modified": sum(1 for x in D["rows"] if x.get("e1_added"))}
        summ["groups"][g] = {"desc": D["meta"]["desc"], "metrics": mets}
    pairs = (a.pairs or "B,C42;B,C43;A,C42;A,B;C42,C43;A,A_E2;C42,C42_E2;B,B_E2;A,A+E1;B,B+E1;C42,C42+E1;C42,D;A,D").split(";")
    for pair in pairs:
        x, y = pair.split(",")
        if x in data and y in data:
            summ["pairs"][f"{x}_vs_{y}"] = {
                k: sv2.paired_cluster([{"rec": r["rec"], "s": {"v": r["s"][key]}} for r in data[x]["rows"]],
                                      [{"rec": r["rec"], "s": {"v": r["s"][key]}} for r in data[y]["rows"]], "v", sel, lower_is_better=low)
                for k, (sel, key, low) in METRICS.items()}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    base_groups = [g for g in data if "+E1" not in g]
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["question_id", "behavior", "subtype", "conflict_kind", "question", "reference_answer", "required_facts"]
        for g in base_groups:
            cols += [f"{g}_answer", f"{g}_content_ok", f"{g}_complete", f"{g}_multi_cov", f"{g}_attrib", f"{g}_whole_refusal",
                     f"{g}_support", f"{g}+E1_content_ok"]
        cols += ["review_status", "human_verdict", "human_note"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(order):
            r = recs[q]
            row = {"question_id": q, "behavior": r["behavior"], "subtype": r["subtype"], "conflict_kind": r.get("conflict_kind"),
                   "question": r["question"], "reference_answer": r["reference_answer"], "required_facts": " / ".join(r["facts"]),
                   "review_status": "provisional", "human_verdict": "", "human_note": ""}
            for g in base_groups:
                s = data[g]["rows"][i]["s"]
                row.update({f"{g}_answer": data[g]["rows"][i]["g"]["answer"], f"{g}_content_ok": int(s["content_ok"]),
                            f"{g}_complete": s["complete"], f"{g}_multi_cov": s["multi_chunk_coverage"], f"{g}_attrib": s["fact_attribution_ok"],
                            f"{g}_whole_refusal": s["whole_refusal_on_answerable"], f"{g}_support": f"{s['n_supported']}/{s['n_claims']}",
                            f"{g}+E1_content_ok": int(data[f"{g}+E1"]["rows"][i]["s"]["content_ok"])})
            w.writerow(row)
    for g in base_groups:
        write_jsonl(out / f"generations_{g}.jsonl", [data[g]["gens"][q] for q in order])
    keys = list(METRICS) + ["claims_supported(按陈述)", "v2_content_ok"]
    shown = [g for g in data if g in base_groups or g in ("A+E1", "B+E1", "C42+E1")]
    print(f"{'指标':30s}" + "".join(f"{g[:10]:>12s}" for g in shown))
    for k in keys:
        print(f"{k:30s}" + "".join(f"{summ['groups'][g]['metrics'][k]['k']}/{summ['groups'][g]['metrics'][k]['n']:<4}".rjust(12) for g in shown))
    for g in base_groups:
        print(g, summ["groups"][g]["metrics"]["cost"])
    for p, M in summ["pairs"].items():
        print(p, {k: (M[k].get("diff"), M[k].get("ci95_cluster_bootstrap")) for k in ("content_ok", "multi_content_complete",
                                                                                         "whole_refusal(应回答类)", "refusal_ok", "claim_support_full(全分母)")})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--name", required=True)
    f.add_argument("--src", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--group", required=True)
    g.add_argument("--input", required=True)
    g.add_argument("--out", required=True)
    n = sub.add_parser("analyze")
    n.add_argument("--input", required=True)
    n.add_argument("--ref", required=True)
    n.add_argument("--runs", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--pairs", default=None)
    a = ap.parse_args(argv)
    {"freeze": cmd_freeze, "gen": cmd_gen, "analyze": cmd_analyze}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
