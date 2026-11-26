"""用本机 qwen2.5:7b 起草问答草稿（原始 JSON 逐条缓存在 data/drafts/，可复现、可审阅）。

    python -m rag_finetuning.experiments.p6_v3.scripts.gen_qa --split train

* single：每个小节 2 道考查不同事实的问题，每条答案拆成“陈述 + 逐字证据原文”；
* multi ：每篇文档 2 道必须同时用到两个小节才能回答的问题。
草稿只是候选：build_v3.py 会做严格校验（证据原文必须是片段子串、数字必须来自证据、必要事实必须出现在证据与答案中），
不通过的直接丢弃。通过校验的样本 generation_method = llm_drafted_auto_checked，review_status = provisional。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_finetuning.common import chunk_document
from rag_finetuning.compat.local_llm import GenerationError, OllamaClient
import time

HERE = Path(__file__).resolve().parents[1]

SINGLE = """你是知识库问答数据标注员。下面是一段制度原文。请就这段原文出 2 道问题，两道题考查的事实不能相同。
要求：
1. 问题像员工真实会问的问题，答案必须能完全由这段原文得到，不需要原文以外的知识。
2. 答案要完整：问题涉及的条件、例外、数值、负责角色都要写全，不能只答一半。
3. 答案拆成若干条陈述 claims；每条陈述的 evidence 必须是从原文中【逐字复制】的连续片段（不得改写），能直接支持这条陈述。
4. required_facts：回答这道题必不可少的 1–4 个关键词或数字，必须逐字出现在原文中（例如“30天”写成原文的“30天”或“30 天”）。
5. topic：用 6–15 个字概括“这道题问的是什么”，例如“退款申请的提交期限”。
只输出 JSON：{"items":[{"question":"…","claims":[{"text":"…","evidence":"…"}],"required_facts":["…"],"topic":"…"}, …]}"""

MULTI = """你是知识库问答数据标注员。下面是同一份制度中的两段原文 A 和 B。请出 1 道必须【同时使用 A 和 B】才能完整回答的问题。
要求：
1. 只用 A 或只用 B 都不能完整回答；答案中至少一条陈述来自 A、至少一条来自 B。
2. 答案要写全涉及的条件、例外、数值和角色。
3. claims 中每条陈述注明 source（"A" 或 "B"），evidence 是从对应原文中【逐字复制】的连续片段。
4. required_facts：1–5 个必不可少的关键词或数字，必须逐字出现在 A 或 B 中。
只输出 JSON：{"question":"…","claims":[{"text":"…","source":"A","evidence":"…"}],"required_facts":["…"]}"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args(argv)
    corpus = HERE / "data" / "corpus" / a.split
    out = HERE / "data" / "drafts" / a.split
    out.mkdir(parents=True, exist_ok=True)
    c = OllamaClient(model="qwen2.5:7b", timeout_s=600)
    info = c.model_info()
    opts = {"temperature": 0, "seed": 42, "num_predict": 900, "num_ctx": 4096}

    def chat(msgs):
        for k in range(4):
            try:
                return c.chat(msgs, opts, fmt="json")
            except GenerationError as e:
                print("retry", k, e.kind, flush=True)
                time.sleep(5 * (k + 1))
        return None   # 连续失败：记录为起草失败并跳过该条（计入数据卡），不阻塞其余文档
    n = 0
    for p in sorted(corpus.glob("*.md")):
        chunks = chunk_document(p, a.split)
        for ch in chunks:
            f = out / f"{p.stem}__s{ch.section}.json"
            if f.exists():
                continue
            g = chat([{"role": "system", "content": SINGLE}, {"role": "user", "content": f"【原文】\n{ch.text}"}])
            f.write_text(json.dumps({"kind": "single", "doc": p.stem, "section": ch.section, "raw": g.text if g else "",
                                     "error": None if g else "ollama_http_500_x4"}, ensure_ascii=False),
                         encoding="utf-8")
            n += 1
        pairs = [(1, 2), (3, 4)] if len(chunks) >= 4 else [(1, 2)]
        if len(chunks) >= 6:
            pairs = [(1, 2), (3, 5), (4, 6)][:2] + [(2, 6)]
        for i, j in pairs[:2]:
            f = out / f"{p.stem}__m{i}_{j}.json"
            if f.exists():
                continue
            A, B = chunks[i - 1], chunks[j - 1]
            g = chat([{"role": "system", "content": MULTI},
                      {"role": "user", "content": f"【原文 A】\n{A.text}\n\n【原文 B】\n{B.text}"}])
            f.write_text(json.dumps({"kind": "multi", "doc": p.stem, "sections": [i, j], "raw": g.text if g else "",
                                     "error": None if g else "ollama_http_500_x4"}, ensure_ascii=False),
                         encoding="utf-8")
            n += 1
        print(p.stem, flush=True)
    (out / "_meta.json").write_text(json.dumps({"model": info, "options": opts, "single_prompt": SINGLE, "multi_prompt": MULTI},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    print("drafts written:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
