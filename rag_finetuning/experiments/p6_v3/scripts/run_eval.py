"""P6-v3 评估：冻结输入 + 统一推理 + 评分器 v2 分析。

    python -m rag_finetuning.experiments.p6_v3.scripts.run_eval freeze --name test_v3 --src <built.jsonl>
    python -m rag_finetuning.experiments.p6_v3.scripts.run_eval gen --group C --input test_v3 --out <dir>
    python -m rag_finetuning.experiments.p6_v3.scripts.run_eval analyze --input test_v3 --ref <built.jsonl> --runs <dir> --out <dir>

所有组：平台 rag_v1.0 Zero-shot Prompt、相同证据与顺序、贪心解码、max_new_tokens=384、batch 1。
HF 组为 bitsandbytes 4-bit NF4；F 组为 Ollama qwen2.5:7b Q4_K_M（temperature 0、seed 42、num_ctx 8192）。
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
from rag_finetuning.compat.rag_pipeline import build_messages

V3 = Path(__file__).resolve().parents[1]
FROZEN = V3 / "data" / "frozen"
RUNS = "workspace/runs"
M15 = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
M3 = ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1")
GROUPS = {
    "A": {"desc": "1.5B 原始模型", "model": M15, "adapter": None},
    "B": {"desc": "1.5B + P6-v1 QLoRA（v1 数据）", "model": M15, "adapter": f"{RUNS}/qlora_r8_v1/adapter"},
    "C": {"desc": "1.5B + v3 数据 QLoRA", "model": M15, "adapter": f"{RUNS}/p6_v3/v3_1p5b_full/adapter"},
    "D": {"desc": "3B 原始模型", "model": M3, "adapter": None},
    "E": {"desc": "3B + v3 数据 QLoRA", "model": M3, "adapter": f"{RUNS}/p6_v3/v3_3b_full/adapter"},
    "F": {"desc": "Ollama qwen2.5:7b（Q4_K_M）规模参考组", "ollama": "qwen2.5:7b"},
    "C_seed43": {"desc": "事后补充：C 的同配置种子 43 复训（估计训练随机性，不替换 C）", "model": M15,
                 "adapter": f"{RUNS}/p6_v3/v3_1p5b_full_seed43/checkpoint_epoch2"},
    "abl_no_partial": {"desc": "消融：v3 去掉部分回答样本", "model": M15, "adapter": f"{RUNS}/p6_v3/abl_1p5b_no_partial/checkpoint_epoch2"},
    "abl_refusal_high": {"desc": "消融：v3 拒答比例提高到 v1 水平", "model": M15, "adapter": f"{RUNS}/p6_v3/abl_1p5b_refusal_high/checkpoint_epoch2"},
    "abl_minus_new_positives": {"desc": "消融：v3 去掉新增有证据正样本", "model": M15,
                                "adapter": f"{RUNS}/p6_v3/abl_1p5b_minus_new_positives/checkpoint_epoch2"},
    "abl_coarse_cite": {"desc": "消融：v3 改为答案末尾粗粒度引用", "model": M15, "adapter": f"{RUNS}/p6_v3/abl_1p5b_coarse_cite/checkpoint_epoch2"},
}
GEN = {"max_new_tokens": 384, "do_sample": False, "repetition_penalty": 1.0}
OLLAMA_OPTS = {"temperature": 0, "seed": 42, "num_predict": 384, "num_ctx": 8192}


def item_hash(r):
    return hashlib.sha256(json.dumps({"question": r["question"], "evidence": [[e["cid"], e["source"], e["text"]] for e in r["evidence"]]},
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def msgs(r):
    return build_messages(r["question"], [(e["cid"], e["text"]) for e in r["evidence"]], {e["cid"]: e["source"] for e in r["evidence"]})


def cmd_freeze(a):
    rows = read_jsonl(Path(a.src))
    refs = [r["reference_answer"].split("\n\n引用：")[0] for r in rows if r["behavior"] != "refuse"]
    for r in rows:
        blob = "\n".join(m["content"] for m in msgs(r))
        assert not any(len(x) > 12 and x in blob for x in refs), f"{r['id']}: 参考答案出现在 Prompt 中"
    frozen = [{"id": r["id"], "question": r["question"], "evidence": [{k: e[k] for k in ("cid", "source", "doc", "text")} for e in r["evidence"]],
               "input_sha256": item_hash(r)} for r in rows]
    FROZEN.mkdir(parents=True, exist_ok=True)
    p = FROZEN / f"{a.name}.jsonl"
    text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in frozen)
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{p} 已存在且内容不同：拒绝覆盖冻结集")
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
    rows = read_jsonl(p)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M15[0], revision=M15[1], local_files_only=True)   # 统一 token 计数口径
    if "ollama" in g:
        from rag_finetuning.compat.local_llm import OllamaClient, gpu_memory_used_mib
        c = OllamaClient(model=g["ollama"], timeout_s=900)
        info = c.model_info()

        def run(m):
            r = c.chat(m, dict(OLLAMA_OPTS))
            return r.text, {"latency_ms": r.total_ms, "output_tokens": r.output_tokens, "ollama_prompt_eval_count": r.prompt_tokens,
                            "gpu_used_total_mib": gpu_memory_used_mib()}
        settings = {"backend": "ollama", **OLLAMA_OPTS, "model": g["ollama"], "model_info": info}
        name = f"ollama:{g['ollama']}"
    else:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        mid, rev = g["model"]
        mtok = AutoTokenizer.from_pretrained(mid, revision=rev, local_files_only=True)
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
            ids = mtok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                o = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=mtok.eos_token_id, **GEN)
            torch.cuda.synchronize()
            new = o[0, ids.shape[1]:]
            return mtok.decode(new, skip_special_tokens=True).strip(), {
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "output_tokens": int(new.shape[0]),
                "gpu_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
        settings = {"backend": "transformers", **GEN, "batch_size": 1, "model_id": mid, "revision": rev, "adapter": g["adapter"],
                    "quantization": "bitsandbytes 4-bit NF4 + double quant, bf16 compute", "load_peak_alloc_mib": load_peak}
        name = f"{mid.split('/')[-1]}" + (f"+{Path(g['adapter']).parent.name}" if g["adapter"] else "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        m = msgs(r)
        ans, extra = run(m)
        res.append({"id": r["id"], "question_id": r["id"], "group": a.group, "model": name, "prompt_variant": "zero_shot",
                    "answer": ans, "input_sha256": r["input_sha256"], "evidence_ids": [e["cid"] for e in r["evidence"]],
                    "input_tokens_qwen_tokenizer": len(tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)), **extra})
        print(f"[{i}/{len(rows)}] {r['id']} {extra['latency_ms']:.0f}ms", flush=True)
    write_jsonl(out / "generations.jsonl", res)
    meta = {"group": a.group, "desc": g["desc"], "model": name, "input": a.input, "frozen_sha256": man["frozen_sha256"],
            "generation_settings": settings, "n": len(res), "wall_s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if "ollama" in g:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:11434/api/generate", method="POST",
                                                      data=json.dumps({"model": g["ollama"], "keep_alive": 0}).encode()), timeout=60).read()
    print("DONE", json.dumps({k: meta[k] for k in ("group", "model", "n", "wall_s")}, ensure_ascii=False))


ANS = ("answer", "partial", "conflict")
METRICS = {   # 名称: (适用题, 取值, 越低越好)
    "content_ok": (lambda r: True, lambda s: s["content_ok"], False),
    "fact_coverage_complete": (lambda r: r["behavior"] in ANS, lambda s: s["complete"], False),
    "false_refusal": (lambda r: r["behavior"] == "answer", lambda s: s["false_refusal"], True),
    "refusal_correct": (lambda r: r["behavior"] == "refuse", lambda s: s["behavior_ok"], False),
    "refusal_correct_strict": (lambda r: r["behavior"] == "refuse", lambda s: s["refusal_strict"], False),
    "partial_correct": (lambda r: r["behavior"] == "partial", lambda s: s["behavior_ok"], False),
    "citation_format_ok": (lambda r: True, lambda s: s["format_ok"], False),
    "citation_valid(no_fabricated)": (lambda r: True, lambda s: not s["fabricated_citations"], False),
    "claim_support_all": (lambda r: r["behavior"] in ANS, lambda s: s["all_claims_supported"], False),
    "multi_evidence_complete": (lambda r: len(r["gold_cids"]) >= 2 and r["behavior"] != "refuse" and not r.get("no_conflict"),
                                lambda s: s["citation_recall"] == 1.0 and s["complete"] is True, False),
    "conflict_detected": (lambda r: r["behavior"] == "conflict", lambda s: s["behavior_ok"], False),
    "no_conflict_handled": (lambda r: bool(r.get("no_conflict")), lambda s: s["behavior_ok"], False),
    "injection_resisted": (lambda r: r.get("subtype") == "prompt_injection",
                           lambda s: s["behavior_ok"] and not s["forbidden_hits"] and not s["fabricated_citations"], False),
}


def cmd_analyze(a):
    man = json.loads((FROZEN / f"{a.input}.manifest.json").read_text(encoding="utf-8"))
    recs = {r["id"]: r for r in read_jsonl(Path(a.ref))}
    order = list(man["items"])
    runs = Path(a.runs)
    groups = [g for g in (a.groups.split(",") if a.groups else GROUPS) if (runs / g / "generations.jsonl").exists()]
    data = {}
    for g in groups:
        gens = {x["id"]: x for x in read_jsonl(runs / g / "generations.jsonl")}
        meta = json.loads((runs / g / "meta.json").read_text(encoding="utf-8"))
        assert meta["frozen_sha256"] == man["frozen_sha256"], g
        assert all(gens[q]["input_sha256"] == man["items"][q] for q in order), g
        data[g] = {"meta": meta, "rows": [{"rec": recs[q], "s": sv2.score_item_v2(recs[q], gens[q]["answer"]), "g": gens[q]} for q in order]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"input": a.input, "n": len(order), "scorer": f"v{sv2.SCORER_VERSION} (support threshold {sv2.SUPPORT_THRESHOLD})",
            "label": "自动规则评分，provisional（未经人工审核）", "groups": {}, "pairs": {}}
    for g, D in data.items():
        mets = {}
        for k, (sel, f, low) in METRICS.items():
            mets[k] = sv2.rate(f(x["s"]) for x in D["rows"] if sel(x["rec"]))
        sup = [x["s"] for x in D["rows"] if x["s"]["n_factual_claims"]]
        mets["claim_support_rate(claims)"] = {"k": sum(s["n_supported_claims"] for s in sup), "n": sum(s["n_factual_claims"] for s in sup)}
        lat = sorted(x["g"]["latency_ms"] for x in D["rows"])
        mets["cost"] = {"latency_ms_p50": statistics.median(lat), "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))],
                        "output_tokens_mean": round(statistics.mean(x["g"].get("output_tokens") or 0 for x in D["rows"]), 1),
                        "input_tokens_mean": round(statistics.mean(x["g"]["input_tokens_qwen_tokenizer"] for x in D["rows"]), 1)}
        summ["groups"][g] = {"desc": D["meta"]["desc"], "model": D["meta"]["model"], "metrics": mets,
                             "review_flags": dict(Counter(f for x in D["rows"] for f in x["s"]["review_flags"])),
                             "generation_settings": D["meta"]["generation_settings"]}
    for pair in (a.pairs or "A,B;A,C;B,C;D,E;E,F;C,E;A,D").split(";"):
        x, y = pair.split(",")
        if x in data and y in data:
            summ["pairs"][f"{x}_vs_{y}"] = {k: sv2.paired_cluster(data[x]["rows"], data[y]["rows"], {
                "content_ok": "content_ok"}.get(k, None) or k, sel, lower_is_better=low) if False else
                sv2.paired_cluster([{"rec": r["rec"], "s": {"v": f(r["s"])}} for r in data[x]["rows"]],
                                   [{"rec": r["rec"], "s": {"v": f(r["s"])}} for r in data[y]["rows"]], "v", sel, lower_is_better=low)
                for k, (sel, f, low) in METRICS.items()}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["question_id", "behavior", "subtype", "no_conflict", "question", "gold_ids", "reference_answer"]
    for g in data:
        cols += [f"{g}_answer", f"{g}_content_ok", f"{g}_complete", f"{g}_false_refusal", f"{g}_format_ok", f"{g}_claims_supported",
                 f"{g}_review_flags"]
    cols += ["review_priority", "human_verdict", "human_note"]
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(order):
            r = recs[q]
            row = {"question_id": q, "behavior": r["behavior"], "subtype": r["subtype"], "no_conflict": r.get("no_conflict"),
                   "question": r["question"], "gold_ids": " ".join(r["gold_cids"]), "reference_answer": r["reference_answer"],
                   "human_verdict": "", "human_note": ""}
            flags = 0
            for g, D in data.items():
                s, gg = D["rows"][i]["s"], D["rows"][i]["g"]
                row.update({f"{g}_answer": gg["answer"], f"{g}_content_ok": int(s["content_ok"]), f"{g}_complete": s["complete"],
                            f"{g}_false_refusal": int(s["false_refusal"]), f"{g}_format_ok": int(s["format_ok"]),
                            f"{g}_claims_supported": f"{s['n_supported_claims']}/{s['n_factual_claims']}",
                            f"{g}_review_flags": ";".join(s["review_flags"])})
                flags += len(s["review_flags"])
            row["review_priority"] = "high" if flags else "normal"
            w.writerow(row)
    # 案例
    cases = {}
    for g, D in data.items():
        cases[g] = {k: [{"id": x["rec"]["id"], "question": x["rec"]["question"], "answer": x["g"]["answer"],
                         "unsupported_claims": [c["text"] for c in x["s"]["claims"] if not c["supported"]]}
                        for x in D["rows"] if cond(x)] for k, cond in {
            "false_refusal": lambda x: x["s"]["false_refusal"],
            "valid_citation_but_unsupported_claim": lambda x: x["s"]["format_ok"] and x["s"]["all_claims_supported"] is False,
            "conflict_failed": lambda x: x["rec"]["behavior"] == "conflict" and not x["s"]["behavior_ok"],
            "no_conflict_misjudged": lambda x: x["rec"].get("no_conflict") and not x["s"]["behavior_ok"],
            "injection_failed": lambda x: x["rec"].get("subtype") == "prompt_injection" and not (x["s"]["behavior_ok"] and not x["s"]["forbidden_hits"]),
            "refusal_failed": lambda x: x["rec"]["behavior"] == "refuse" and not x["s"]["behavior_ok"],
            "partial_failed": lambda x: x["rec"]["behavior"] == "partial" and not x["s"]["behavior_ok"],
        }.items()}
    (out / "error_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    for g, D in data.items():
        write_jsonl(out / f"generations_{g}.jsonl", [x["g"] for x in D["rows"]])
    keys = list(METRICS) + ["claim_support_rate(claims)"]
    print(f"{'指标':34s}" + "".join(f"{g[:14]:>16s}" for g in data))
    for k in keys:
        cells = []
        for g in data:
            m = summ["groups"][g]["metrics"][k]
            cells.append(f"{m['k']}/{m['n']}")
        print(f"{k:34s}" + "".join(f"{c:>16s}" for c in cells))
    for g in data:
        print(g, summ["groups"][g]["metrics"]["cost"])
    for p, M in summ["pairs"].items():
        print(p, {k: (M[k].get("diff"), M[k].get("ci95_cluster_bootstrap")) for k in ("content_ok", "fact_coverage_complete", "false_refusal",
                                                                                         "claim_support_all", "conflict_detected")})


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
    n.add_argument("--groups", default=None)
    n.add_argument("--pairs", default=None)
    a = ap.parse_args(argv)
    {"freeze": cmd_freeze, "gen": cmd_gen, "analyze": cmd_analyze}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
