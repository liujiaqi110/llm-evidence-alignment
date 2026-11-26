"""P6-v5 偏好数据重建（evidence-first）。只使用 P6-v1 / v3 / v4 训练语料；不读取 hard_dev / test_v5 的任何标注。

    python -m rag_finetuning.experiments.p6_v5.scripts.build_pref_ef select      # 程序选片段组合 + 抽必要事实 + 生成参考答案
    python -m rag_finetuning.experiments.p6_v5.scripts.build_pref_ef rewrite     # 7B 只把模板问题改写成自然问法（Ollama）
    python -m rag_finetuning.experiments.p6_v5.scripts.build_pref_ef assemble    # 校验改写问题、组装证据、伪多证据检查、隔离检查
    python -m rag_finetuning.experiments.p6_v5.scripts.build_pref_ef candidates --group A|B|C42   # 3B 真实回答
    python -m rag_finetuning.experiments.p6_v5.scripts.build_pref_ef pairs       # 偏好对 + 审计 + 长度控制

evidence-first 流程（自然语料）：
  1 程序在同一文档中选 2～3 个小节；
  2 程序从每个小节原文中抽“必要事实”（数量 + 单位、审批 / 负责角色、禁止 / 不予等限制词），只保留该小节独有、其他所选小节没有的事实；
     每个小节至少 1 条，否则放弃该组合 —— 因此任何单个片段都不含全部必要事实（伪多证据样本被排除）；
  3 参考答案 = 每条必要事实所在的原文句子（原样摘录）+ 对应片段 ID，由构造保证正确；
  4 7B 只做一件事：把程序生成的模板问题改写成员工自然的问法（看得到必要事实，但不得在问题中写出它们）；
  5 改写问题校验：不含任何必要事实、长度合适、与每个所选小节都有内容重合（否则判为“问题不需要该小节”）。
结构化语料：直接复用 P6-v4 的 324 条程序化样本（答案由构造保证正确）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from rag_finetuning.common import REPO, read_jsonl, write_jsonl
from rag_finetuning.evaluation.scoring import CITE_RE, normalize, numbers
from rag_finetuning.experiments.p6_v4.scripts.eng import QTY, relevance
from rag_finetuning.experiments.p6_v5.scripts.build_pref_v5 import CANON_REFUSAL, construct, eval_docs, natural_chunks, sentences_with_cites, with_cite_line
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import score_item_v4
from rag_finetuning.scripts import build_dataset as bd

V5 = Path(__file__).resolve().parents[1]
EXP = V5.parent
OUT = V5 / "data" / "train_pref"
# 1.0（选择时使用，已存档）：角色前缀过长，会抽出“员工档案需经部门经理”这类片段 → 模型同义改写时被误判为漏答
ROLE_V10 = re.compile(r"([一-龥]{1,8}(?:部门负责人|负责人|总监|经理|主管|委员会|部|组|中心|办公室))(?:审批|审核|批准|确认|备案|同意|负责)")
LIMIT_V10 = re.compile(r"(不得[一-龥]{1,8}|禁止[一-龥]{1,8}|不予[一-龥]{1,6}|严禁[一-龥]{1,8})")
# 1.1（组装时重新推导必要事实）：只取“经 / 由 / 向 / 交 / 报”之后的角色名本身；限制词只取 2～4 个字
ROLE = re.compile(r"(?<=[经由向交])((?:(?![经由需须向交报并和及或的，。；、])[A-Za-z一-龥]){0,6}(?:部门负责人|负责人|总经理|总监|经理|主管|委员会|部|组|中心|办公室))")
LIMIT = re.compile(r"((?:不得|禁止|严禁)[一-龥]{2,4}|不予[一-龥]{2,4})")
REL_MIN = 0.12

REWRITE_SYS = "你负责把企业制度问答的模板问题改写成员工真实会问的问题。不要回答问题，不要编造制度内容。"
REWRITE_USER = """制度文档《{title}》中的以下小节：
{heads}

回答这个问题必须同时用到下列要点（这些要点只是让你知道问题要覆盖什么，不能写进问题里）：
{facts}

模板问题：{tmpl}

请把模板问题改写成一句自然、口语化的员工提问，要求：问题要让人必须查看上面全部小节才能答完整；不要在问题里写出任何要点中的数字、部门、角色或期限；不超过 50 个字。
只输出改写后的问题本身。"""


def chunk_facts(c, others_text: str):
    body = c.text.split("\n", 1)[-1]
    sents = [s for s in re.split(r"(?<=[。；])", body) if s.strip()]
    out = []
    for s in sents:
        cands = [m.group(0).replace(" ", "") for m in QTY.finditer(s)] + [m.group(1) for m in ROLE.finditer(s)] + [m.group(1) for m in LIMIT.finditer(s)]
        for f in cands:
            nf = normalize(f)
            if len(nf) < 2 or nf in normalize(others_text) or any(nf == normalize(x["fact"]) for x in out):
                continue
            out.append({"fact": f, "sentence": s.strip()})
    return out[:2]


def cmd_select(a):
    chunks = natural_chunks()
    names, texts, _ = eval_docs()
    assert not ({c.doc for c in chunks.values()} & names) and not ({c.text for c in chunks.values()} & texts), "训练语料与评测语料重叠"
    by_doc = defaultdict(list)
    for k, c in chunks.items():
        by_doc[c.doc].append(c)
    rng = random.Random(20261124)
    items, stat = [], Counter()
    for doc, cs in sorted(by_doc.items()):
        cs = sorted(cs, key=lambda x: x.section)
        combos = [list(x) for x in combinations(cs, 2)] + [list(x) for x in combinations(cs, 3)]
        rng.shuffle(combos)
        got = 0
        for grp in combos:
            if got >= 6:
                break
            stat["组合尝试"] += 1
            facts = []
            for c in grp:
                others = " ".join(o.text for o in grp if o is not c)
                fs = chunk_facts(c, others)
                if not fs:
                    break
                facts.append((c, fs))
            if len(facts) != len(grp):
                stat["有小节抽不出独有必要事实"] += 1
                continue
            # 伪多证据检查：任何单个片段都不能包含全部必要事实
            allf = [f["fact"] for _, fs in facts for f in fs]
            if any(all(normalize(f) in normalize(c.text) for f in allf) for c in grp):
                stat["单个片段已足以回答（伪多证据）"] += 1
                continue
            title = grp[0].text.split(" > ")[0]
            heads = [c.text.split("\n")[0].split(" > ")[-1] for c in grp]
            tmpl = f"关于{title}，" + "、".join(f"“{h}”" for h in heads) + "这几方面分别有什么规定？"
            parts = []
            for c, fs in facts:                       # 按片段顺序，每个片段内去重句子
                for sent in dict.fromkeys(f["sentence"] for f in fs):
                    parts.append(f"{sent.rstrip('。；')}[{c.cid}]。")
            ans = "".join(parts)
            items.append({"id": f"v5ef_{len(items):04d}", "doc": doc, "keys": [f"{c.doc}#{c.section}" for c in grp], "split": grp[0].split,
                          "title": title, "heads": heads, "template_question": tmpl,
                          "facts": [f["fact"] for _, fs in facts for f in fs], "fact_cids": [c.cid for c, fs in facts for f in fs],
                          "fact_sentences": [f["sentence"] for _, fs in facts for f in fs], "reference_body": ans})
            got += 1
            stat["选中"] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "ef_selected.jsonl", items)
    (OUT / "ef_select_report.json").write_text(json.dumps({"n_docs": len(by_doc), **stat}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_docs": len(by_doc), **stat}, ensure_ascii=False))


def cmd_rewrite(a):
    from rag_finetuning.compat.local_llm import OllamaClient
    c = OllamaClient(model="qwen2.5:7b", base_url=os.environ.get("P6_OLLAMA_URL", "http://127.0.0.1:11434"), timeout_s=300)
    info = c.model_info()
    rows = read_jsonl(OUT / "ef_selected.jsonl")
    out = []
    for i, r in enumerate(rows, 1):
        m = [{"role": "system", "content": REWRITE_SYS},
             {"role": "user", "content": REWRITE_USER.format(title=r["title"], heads="\n".join(f"- {h}" for h in r["heads"]),
                                                             facts="\n".join(f"- {f}" for f in r["facts"]), tmpl=r["template_question"])}]
        x = c.chat(m, {"temperature": 0.7, "seed": 2000 + i, "num_predict": 96, "num_ctx": 4096})
        out.append({"id": r["id"], "raw": x.text, "latency_ms": x.total_ms})
        if i % 20 == 0:
            print(f"[{i}/{len(rows)}]", flush=True)
    write_jsonl(OUT / "ef_rewrites_raw.jsonl", out)
    (OUT / "ef_rewrite_meta.json").write_text(json.dumps({"model": "ollama:qwen2.5:7b", "model_info": info, "temperature": 0.7, "seed_base": 2000,
                                                         "n": len(out)}, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_assemble(a):
    chunks = natural_chunks()
    by_cid = {c.cid: c for c in chunks.values()}
    names, texts, eval_q = eval_docs()
    sel = {r["id"]: r for r in read_jsonl(OUT / "ef_selected.jsonl")}
    rows, rej, seen = [], Counter(), []
    for x in read_jsonl(OUT / "ef_rewrites_raw.jsonl"):
        r = dict(sel[x["id"]])
        # 1.1：用收紧后的规则重新推导必要事实与参考答案（问题与证据不变）
        grp0 = [chunks[k] for k in r["keys"]]
        refs = []
        for c in grp0:
            refs.append((c, chunk_facts(c, " ".join(o.text for o in grp0 if o is not c))))
        if any(not fs for _, fs in refs):
            rej["收紧规则后有小节没有独有必要事实"] += 1
            continue
        r["facts"] = [f["fact"] for _, fs in refs for f in fs]
        r["fact_cids"] = [c.cid for c, fs in refs for f in fs]
        r["reference_body"] = "".join(f"{sent.rstrip('。；')}[{c.cid}]。" for c, fs in refs for sent in dict.fromkeys(f["sentence"] for f in fs))
        if any(all(normalize(f) in normalize(c.text) for f in r["facts"]) for c in grp0):
            rej["单个片段已足以回答（伪多证据）"] += 1
            continue
        q = re.sub(r"^(改写后的问题[:：]\s*|问题[:：]\s*)", "", x["raw"].strip().splitlines()[0].strip()).strip("“”\"' ")
        grp = [chunks[k] for k in r["keys"]]
        why = None
        if not (8 <= len(q) <= 60) or not q.endswith(("？", "?")):
            why = "改写问题长度或形式不合格"
        elif any(normalize(f) in normalize(q) for f in r["facts"]):
            why = "改写问题中出现了必要事实"
        elif any(relevance(q, c.text) < REL_MIN for c in grp):
            why = "改写问题与某个所选小节没有内容重合（问题不需要该小节）"
        elif max([bd.jaccard(q, s) for s in seen] or [0]) > 0.8:
            why = "与已有问题近重复"
        elif max(bd.jaccard(q, s) for s in eval_q) > 0.5:
            why = "与评测集 / 旧测试集问题过于相似"
        if why:
            rej[why] += 1
            continue
        item = {"id": r["id"], "type": "B", "behavior": "answer", "question": q, "gold": r["keys"], "facts": r["facts"],
                "answer": "".join("[{%d}]" % (i + 1) for i in range(len(grp)))}
        rec = bd.build_record(item, grp[0].split, {k: v for k, v in chunks.items() if v.split == grp[0].split})
        ref = with_cite_line(r["reference_body"])
        rec.update({"reference_answer": ref, "facts": r["facts"], "fact_cids": r["fact_cids"], "subtype": "natural_multi_ef",
                    "generation_method": "evidence_first（程序选片段与事实，7B 只改写问法）", "review_status": "provisional",
                    "dataset_version": "p6v5-pref-1.0", "no_conflict": False, "conflict_kind": None, "template_question": r["template_question"]})
        s = score_item_v4(rec, ref)
        if not (s["content_ok_v4"] and s["format_ok"] and s["multi_complete_v5"] and s["fact_attribution_ok"] is not False and s["claim_support_full"]):
            rej["参考答案未通过评分器自检"] += 1
            continue
        if any(all(normalize(f) in normalize(e["text"]) for f in r["facts"]) for e in rec["evidence"]):
            rej["组装后单个证据片段已含全部必要事实（伪多证据）"] += 1
            continue
        seen.append(q)
        rows.append(rec)
    write_jsonl(OUT / "natural_ef.jsonl", rows)
    st = [dict(r, origin="v4_structured") for r in read_jsonl(EXP / "p6_v4/data/v4/structured_only/train.jsonl")]
    write_jsonl(OUT / "structured.jsonl", st)
    rep = {"selected": len(sel), "rewrites": sum(1 for _ in read_jsonl(OUT / "ef_rewrites_raw.jsonl")), "accepted_natural": len(rows),
           "rejected": dict(rej), "natural_docs": len({r["source_docs"][0] for r in rows}), "structured": len(st),
           "gold_chunks_per_item": dict(Counter(len(r["gold_cids"]) for r in rows))}
    (OUT / "assemble_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


def cmd_candidates(a):
    from rag_finetuning.experiments.p6_v5.scripts.run_eval_v5 import GROUPS, make_chat, msgs
    chat, settings, name, _ = make_chat(GROUPS[a.group])
    rows = read_jsonl(OUT / "natural_ef.jsonl") + (read_jsonl(OUT / "structured.jsonl") if a.group != "C42" else [])
    out = []
    for i, r in enumerate(rows, 1):
        x = chat(msgs(r), 384)
        out.append({"id": r["id"], "group": a.group, "model": name, "answer": x["text"], "output_tokens": x["output_tokens"]})
        if i % 25 == 0:
            print(f"[{i}/{len(rows)}]", flush=True)
    write_jsonl(OUT / f"candidates_{a.group}.jsonl", out)
    (OUT / f"candidates_{a.group}.meta.json").write_text(json.dumps({"group": a.group, "model": name, "settings": settings, "n": len(out),
                                                                     "note": "v4（C42）在结构化样本上训练过，所以只在自然语料上生成"},
                                                                    ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("select", "rewrite", "assemble"):
        sub.add_parser(n)
    c = sub.add_parser("candidates")
    c.add_argument("--group", required=True, choices=["A", "B", "C42"])
    a = ap.parse_args()
    {"select": cmd_select, "rewrite": cmd_rewrite, "assemble": cmd_assemble, "candidates": cmd_candidates}[a.cmd](a)
