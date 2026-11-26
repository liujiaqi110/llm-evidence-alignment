"""P6-v5 评估：冻结输入 / 统一推理（含证据覆盖感知生成）/ 评分器 v4 分析。

    python -m rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 freeze --name hard_dev --src <built.jsonl>
    python -m rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 gen --group A --input hard_dev --out <runs>/p6_v5/eval/hard_dev/A
    python -m rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 analyze --input hard_dev --ref <built.jsonl> --runs <dir> --out <out>

规则：
* 所有组：平台 rag_v1.0 Prompt、相同证据与顺序、贪心解码；3B 用 transformers 4-bit NF4，7B 用 Ollama（temperature 0, seed 42）。
* *_plan 组：evidence_planning.run_plan（抽取 → 生成 → 检查 / 必要时修订），权重与对应基础组完全相同。
* 测试集（名字以 test 开头）每组只允许生成一次：已有 meta.json 时拒绝覆盖；开发集可加 --allow-overwrite。
"""
from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from rag_finetuning.common import read_jsonl, write_jsonl
from rag_finetuning.experiments.p6_v3.scripts import scoring_v2 as sv2
from rag_finetuning.experiments.p6_v4.scripts.run_eval_v4 import GEN, M3, M15, OLLAMA_OPTS, item_hash
from rag_finetuning.experiments.p6_v5.scripts.evidence_planning import run_plan
from rag_finetuning.experiments.p6_v5.scripts.planning_v2 import run_planning_v2
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4
from rag_finetuning.compat.rag_pipeline import build_messages

V5 = Path(__file__).resolve().parents[1]
FROZEN = V5 / "data" / "frozen"
RUNS = os.environ.get("P6_RUNS", "workspace/runs")
OLLAMA_URL = os.environ.get("P6_OLLAMA_URL", "http://127.0.0.1:11434")
GROUPS = {
    "A": {"desc": "3B 原始模型（单阶段）", "model": M3, "adapter": None},
    "B": {"desc": "3B + v3 QLoRA", "model": M3, "adapter": f"{RUNS}/p6_v3/v3_3b_full/adapter"},
    "C42": {"desc": "3B + v4 QLoRA 种子 42", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s42/adapter"},
    "C43": {"desc": "3B + v4 QLoRA 种子 43", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s43/adapter"},
    "S42": {"desc": "3B + v5 针对性 SFT 种子 42", "model": M3, "adapter": f"{RUNS}/p6_v5/sft_3b_s42/adapter"},
    "S43": {"desc": "3B + v5 针对性 SFT 种子 43", "model": M3, "adapter": f"{RUNS}/p6_v5/sft_3b_s43/adapter"},
    "P42": {"desc": "3B + v5 DPO（初始化自 S42）种子 42", "model": M3, "adapter": f"{RUNS}/p6_v5/dpo_3b_s42/adapter"},
    "P43": {"desc": "3B + v5 DPO（初始化自 S43）种子 43", "model": M3, "adapter": f"{RUNS}/p6_v5/dpo_3b_s43/adapter"},
    "D": {"desc": "Ollama qwen2.5:7b（Q4_K_M）", "ollama": "qwen2.5:7b"},
    "A_plan": {"desc": "3B 原始模型 + 证据覆盖感知生成", "model": M3, "adapter": None, "plan": True},
    "D_plan": {"desc": "7B + 证据覆盖感知生成", "ollama": "qwen2.5:7b", "plan": True},
    "C42_plan": {"desc": "3B v4 种子 42 + 现有证据覆盖生成（evidence_planning v1）", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s42/adapter", "plan": True},
    "C42_pv2": {"desc": "3B v4 种子 42 + Planning v2（逐片段抽取 → 代码整合 → 生成与核验）", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s42/adapter", "plan": "pv2"},
    "D_pv2": {"desc": "7B + Planning v2（逐片段抽取 → 代码整合 → 生成与核验）", "ollama": "qwen2.5:7b", "plan": "pv2"},
    "C42_pv21": {"desc": "3B v4 种子 42 + Planning v2.1（hard_dev 诊断后修复版）", "model": M3, "adapter": f"{RUNS}/p6_v4/v4_3b_s42/adapter", "plan": "pv21"},
    "D_pv21": {"desc": "7B + Planning v2.1（hard_dev 诊断后修复版）", "ollama": "qwen2.5:7b", "plan": "pv21"},
    "A_plan2": {"desc": "3B 原始模型 + 证据覆盖感知生成 v2（相关性过滤 + 代码补充）", "model": M3, "adapter": None, "plan": "v2"},
    "D_plan2": {"desc": "7B + 证据覆盖感知生成 v2（相关性过滤 + 代码补充）", "ollama": "qwen2.5:7b", "plan": "v2"},
}


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
        raise SystemExit(f"{p} 已存在且内容不同：拒绝覆盖")
    p.write_text(text, encoding="utf-8", newline="\n")
    man = {"name": a.name, "source": a.src, "source_sha256": hashlib.sha256(Path(a.src).read_bytes()).hexdigest(),
           "frozen_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n": len(frozen), "contains_reference_answers": False,
           "items": {x["id"]: x["input_sha256"] for x in frozen}}
    (FROZEN / f"{a.name}.manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("name", "n", "frozen_sha256")}, ensure_ascii=False))


def make_chat(g):
    """返回 chat(messages, max_new_tokens) -> dict 与设置说明。"""
    from transformers import AutoTokenizer
    tok15 = AutoTokenizer.from_pretrained(M15[0], revision=M15[1], local_files_only=True)

    def ntok(m):
        return len(tok15.apply_chat_template(m, add_generation_prompt=True, tokenize=True))
    if "ollama" in g:
        from rag_finetuning.compat.local_llm import OllamaClient
        c = OllamaClient(model=g["ollama"], base_url=OLLAMA_URL, timeout_s=900)
        info = c.model_info()

        def chat(m, n):
            r = c.chat(m, dict(OLLAMA_OPTS, num_predict=n))
            return {"text": r.text, "latency_ms": r.total_ms, "output_tokens": r.output_tokens, "input_tokens": ntok(m)}
        return chat, {"backend": "ollama", **OLLAMA_OPTS, "model_info": info}, f"ollama:{g['ollama']}", c
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

    def chat(m, n):
        ids = tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            o = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id, **dict(GEN, max_new_tokens=n))
        torch.cuda.synchronize()
        new = o[0, ids.shape[1]:]
        return {"text": tok.decode(new, skip_special_tokens=True).strip(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "output_tokens": int(new.shape[0]), "input_tokens": int(ids.shape[1])}
    settings = {"backend": "transformers", **GEN, "model_id": mid, "revision": rev, "adapter": g["adapter"],
                "quantization": "bitsandbytes 4-bit NF4 + double quant, bf16 compute", "load_peak_alloc_mib": load_peak}
    if g["adapter"]:
        settings["adapter_sha256"] = hashlib.sha256((Path(g["adapter"]) / "adapter_model.safetensors").read_bytes()).hexdigest()
    return chat, settings, mid.split("/")[-1] + (f"+{Path(g['adapter']).parent.name}" if g["adapter"] else ""), None


def cmd_gen(a):
    g = GROUPS[a.group]
    p = FROZEN / f"{a.input}.jsonl"
    man = json.loads((FROZEN / f"{a.input}.manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(p.read_bytes()).hexdigest() != man["frozen_sha256"]:
        raise SystemExit("冻结输入哈希不一致")
    out = Path(a.out)
    if (out / "meta.json").exists() and (a.input.startswith("test") or not a.allow_overwrite):
        raise SystemExit(f"{out} 已有结果：测试集每组只允许生成一次（开发集可加 --allow-overwrite）")
    rows = read_jsonl(p)
    chat, settings, name, oll = make_chat(g)
    out.mkdir(parents=True, exist_ok=True)
    # 中断恢复（2026-11-25 test_v5 正式评测前加入；不改变 Prompt / 解码 / 评分）：
    # 每题生成后立即追加到 generations.partial.jsonl；重启时读取已完成的题（核对组名与输入哈希），只生成剩余题，并记录恢复事件。
    part = out / "generations.partial.jsonl"
    done = {}
    if part.exists():
        for x in read_jsonl(part):
            src = next(r for r in rows if r["id"] == x["id"])
            assert x["group"] == a.group and x["input_sha256"] == src["input_sha256"], x["id"]
            done[x["id"]] = x
    resume_log = out / "resume_log.jsonl"
    with open(resume_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "time": time.strftime("%F %T"), "already_done": len(done), "remaining": len(rows) - len(done),
                            "reason": os.environ.get("P6_RESUME_REASON", "")}, ensure_ascii=False) + "\n")
    pf = open(part, "a", encoding="utf-8")
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        if r["id"] in done:
            res.append(done[r["id"]])
            continue
        if g.get("plan") in ("pv2", "pv21"):
            # 推理只读取冻结输入中的问题与证据（cid / source / doc / text），不读取任何标注字段
            pr = run_planning_v2(chat, r["question"], [{k: e[k] for k in ("cid", "source", "doc", "text")} for e in r["evidence"]],
                                 version="2.1" if g["plan"] == "pv21" else "2.0")
            rec = {"answer": pr["answer"], "latency_ms": pr["latency_ms"], "output_tokens": pr["output_tokens"], "input_tokens": pr["input_tokens"],
                   "n_calls": pr["n_calls"], "plan": {k: pr[k] for k in ("facts", "dropped", "consistency_notes", "check", "calls")}}
        elif g.get("plan"):
            pr = run_plan(chat, r["question"], r["evidence"], variant="v2" if g["plan"] == "v2" else "v1")
            rec = {"answer": pr["answer"], "latency_ms": pr["latency_ms"], "output_tokens": pr["output_tokens"], "input_tokens": pr["input_tokens"],
                   "n_calls": pr["n_calls"], "plan": {k: pr[k] for k in ("facts", "dropped", "check", "calls")}}
        else:
            x = chat(msgs(r), GEN["max_new_tokens"])
            rec = {"answer": x["text"], "latency_ms": x["latency_ms"], "output_tokens": x["output_tokens"], "input_tokens": x["input_tokens"], "n_calls": 1}
        res.append({"id": r["id"], "group": a.group, "model": name, "input_sha256": r["input_sha256"], **rec})
        pf.write(json.dumps(res[-1], ensure_ascii=False) + "\n")
        pf.flush()
        print(f"[{i}/{len(rows)}] {r['id']} {rec['latency_ms']:.0f}ms calls={rec['n_calls']}", flush=True)
    pf.close()
    write_jsonl(out / "generations.jsonl", res)
    if torch is not None and torch.cuda.is_available() and "ollama" not in g:
        settings["gen_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        settings["gen_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    meta = {"group": a.group, "desc": g["desc"], "model": name, "input": a.input, "frozen_sha256": man["frozen_sha256"],
            "generation_settings": settings, "plan": bool(g.get("plan")), "n": len(res), "wall_s": round(time.time() - t0, 1),
            "resumed_items": len(done), "resume_log": [json.loads(x) for x in resume_log.read_text(encoding="utf-8").splitlines() if x.strip()]}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if oll is not None:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(f"{OLLAMA_URL}/api/generate", method="POST",
                                                      data=json.dumps({"model": g["ollama"], "keep_alive": 0}).encode()), timeout=60).read()
    print("DONE", json.dumps({k: meta[k] for k in ("group", "model", "n", "wall_s")}, ensure_ascii=False))


ANS = ("answer", "partial", "conflict")
METRICS = {  # 名称: (适用题, 取值键, 越低越好)
    "content_ok_v4（内容正确，不含引用格式）": (lambda r: True, "content_ok_v4", False),
    "content_ok（v3 口径，含引用有效性）": (lambda r: True, "content_ok", False),
    "required_fact_coverage（必要事实覆盖率，均值）": (lambda r: r["behavior"] != "refuse", "required_fact_coverage", False),
    "multi_complete_v5（多证据完整率，主指标）": (lambda r: True, "multi_complete_v5", False),
    "multi_complete（v3 口径）": (lambda r: True, "multi_complete", False),
    "multi_chunk_coverage_v5": (lambda r: True, "multi_chunk_coverage_v5", False),
    "refusal_ok（正确拒答）": (lambda r: r["behavior"] == "refuse", "refusal_ok", False),
    "false_refusal（answer 题错误拒答）": (lambda r: r["behavior"] == "answer", "false_refusal", True),
    "whole_refusal（应回答类整体拒答）": (lambda r: r["behavior"] in ANS, "whole_refusal_on_answerable", True),
    "partial_ok": (lambda r: r["behavior"] == "partial", "partial_ok", False),
    "conflict_true_ok（真冲突识别）": (lambda r: r["behavior"] == "conflict", "conflict_true_ok", False),
    "no_conflict_ok（非冲突判断）": (lambda r: bool(r.get("no_conflict")), "no_conflict_ok", False),
    "claim_support_full（陈述支持，全分母）": (lambda r: r["behavior"] in ANS, "claim_support_full", False),
    "citation_accuracy（事实引用正确）": (lambda r: r["behavior"] in ANS, "citation_accuracy", False),
    "citation_valid（引用 ID 合法）": (lambda r: True, "citation_valid", False),
    "format_ok": (lambda r: True, "format_ok", False),
    "injection_resisted": (lambda r: r.get("subtype") == "prompt_injection", "injection_resisted", False),
    "first_evidence_only（多证据题只用第一个）": (lambda r: True, "first_evidence_only_m", True),
}


def mean_rate(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"k": 0, "n": 0, "rate": None}
    if all(isinstance(x, bool) for x in v):
        return sv2.rate(v)
    return {"k": round(sum(v), 2), "n": len(v), "rate": round(sum(v) / len(v), 4)}


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
        rows = []
        for q in order:
            s = score_item_v4(recs[q], gens[q]["answer"])
            s["first_evidence_only_m"] = s["first_evidence_only"] if s["is_multi_v5"] else None
            rows.append({"rec": recs[q], "s": s, "g": gens[q]})
        data[g] = {"meta": meta, "rows": rows}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = {"input": a.input, "n": len(order), "scorer": f"v{SCORER_VERSION}", "label": "自动规则评分，provisional（未经人工审核）",
            "groups": {}, "pairs": {}}
    for g, D in data.items():
        mets = {k: mean_rate(x["s"][key] for x in D["rows"] if sel(x["rec"])) for k, (sel, key, low) in METRICS.items()}
        lat = sorted(x["g"].get("latency_ms") or 0 for x in D["rows"])
        mets["cost"] = {"latency_ms_p50": statistics.median(lat), "latency_ms_p95": lat[int(0.95 * (len(lat) - 1))],
                        "output_tokens_mean": round(statistics.mean(x["g"].get("output_tokens") or 0 for x in D["rows"]), 1),
                        "input_tokens_mean": round(statistics.mean(x["g"].get("input_tokens") or 0 for x in D["rows"]), 1),
                        "calls_mean": round(statistics.mean(x["g"].get("n_calls") or 1 for x in D["rows"]), 2),
                        "revised": sum(1 for x in D["rows"] if (x["g"].get("plan") or {}).get("check", {}).get("revised"))}
        summ["groups"][g] = {"desc": D["meta"]["desc"], "metrics": mets}
    pairs = [p.split(",") for p in (a.pairs or "").split(";") if p] or [(x, y) for i, x in enumerate(groups) for y in groups[i + 1:]]
    for x, y in pairs:
        if x in data and y in data:
            summ["pairs"][f"{x}_vs_{y}"] = {
                k: sv2.paired_cluster([{"rec": r["rec"], "s": {"v": r["s"][key]}} for r in data[x]["rows"]],
                                      [{"rec": r["rec"], "s": {"v": r["s"][key]}} for r in data[y]["rows"]], "v", sel, lower_is_better=low)
                for k, (sel, key, low) in METRICS.items() if key not in ("required_fact_coverage",)}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "per_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        cols = ["question_id", "behavior", "subtype", "conflict_kind", "tags", "gold_positions", "order_pair", "question", "reference_answer",
                "required_facts"]
        for g in groups:
            cols += [f"{g}_answer", f"{g}_content_ok_v4", f"{g}_fact_cov", f"{g}_multi_v5", f"{g}_missing", f"{g}_support", f"{g}_cite_acc"]
        cols += ["review_status", "human_verdict", "human_note"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, q in enumerate(order):
            r = recs[q]
            row = {"question_id": q, "behavior": r["behavior"], "subtype": r.get("subtype"), "conflict_kind": r.get("conflict_kind"),
                   "tags": ",".join(r.get("hard_tags") or []), "gold_positions": r.get("gold_positions"), "order_pair": r.get("order_pair"),
                   "question": r["question"], "reference_answer": r["reference_answer"], "required_facts": " / ".join(r["facts"]),
                   "review_status": "provisional", "human_verdict": "", "human_note": ""}
            for g in groups:
                s = data[g]["rows"][i]["s"]
                row.update({f"{g}_answer": data[g]["rows"][i]["g"]["answer"], f"{g}_content_ok_v4": int(s["content_ok_v4"]),
                            f"{g}_fact_cov": s["required_fact_coverage"], f"{g}_multi_v5": s["multi_complete_v5"],
                            f"{g}_missing": " / ".join(s["missing_facts"]), f"{g}_support": f"{s['n_supported']}/{s['n_claims']}",
                            f"{g}_cite_acc": s["citation_accuracy"]})
            w.writerow(row)
    with open(out / "per_fact.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "question_id", "fact", "support_cid", "evidence_pos", "hit"])
        for g in groups:
            for x in data[g]["rows"]:
                for d in x["s"]["fact_detail"]:
                    w.writerow([g, x["rec"]["id"], d["fact"], d["cid"], d["evidence_pos"], int(d["hit"])])
    for g in groups:
        write_jsonl(out / f"generations_{g}.jsonl", [x["g"] for x in data[g]["rows"]])
    keys = list(METRICS)
    print(f"{'指标':44s}" + "".join(f"{g[:8]:>10s}" for g in groups))
    for k in keys:
        cells = []
        for g in groups:
            m = summ["groups"][g]["metrics"][k]
            cells.append((f"{m['rate']:.2f}" if isinstance(m["k"], float) else f"{m['k']}/{m['n']}").rjust(10))
        print(f"{k:44s}" + "".join(cells))
    for g in groups:
        print(g, summ["groups"][g]["metrics"]["cost"])
    return data


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
    g.add_argument("--allow-overwrite", action="store_true")
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
