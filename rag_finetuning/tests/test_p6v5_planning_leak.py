"""P6-v5 Planning v2：防 Oracle 信息泄漏的自动测试（不需要 GPU / 模型）。

保证推理阶段的任何 Prompt 都不包含：gold required_facts、gold evidence IDs（作为标注）、reference_answer、
人工标注的事实清单、Oracle 实验的标准事实列表。
"""
import inspect
import json
import unittest
from pathlib import Path

from rag_finetuning.experiments.p6_v5.scripts import planning_v2 as pv2
from rag_finetuning.experiments.p6_v5.scripts import run_eval_v5

V5 = Path(__file__).resolve().parents[1] / "experiments" / "p6_v5"
FORBIDDEN_KEYS = {"facts", "fact_cids", "gold_cids", "reference_answer", "required_facts", "gold", "answer", "forbid", "behavior",
                  "subtype", "conflict_kind", "no_conflict", "hard_tags", "gold_positions", "order_pair"}
SENT_FACT, SENT_REF = "哨兵事实XQZ", "哨兵参考答案QWX"
EV = [{"cid": "aaaaaaaaaa01", "source": "x.md", "doc": "x", "text": "制度 > 申请\n申请须提前 3 个工作日提交，经部门负责人审批。"},
      {"cid": "aaaaaaaaaa02", "source": "x.md", "doc": "x", "text": "制度 > 金额\n金额超过 5000 元的，还须经财务部审批。"}]


from rag_finetuning.tests._private_data import requires  # noqa: E402

class FakeChat:
    def __init__(self):
        self.prompts = []

    def __call__(self, messages, n):
        self.prompts.append("\n".join(m["content"] for m in messages))
        if "资料片段 [" in messages[-1]["content"]:
            cid = messages[-1]["content"].split("资料片段 [", 1)[1][:12]
            txt = {"aaaaaaaaaa01": "申请须提前 3 个工作日提交，经部门负责人审批", "aaaaaaaaaa02": "金额超过 5000 元的，还须经财务部审批"}[cid]
            out = json.dumps([{"fact_text": txt, "conditions": "无", "version": "无", "negation_or_exception": False, "necessity": "必要",
                               "support_span": txt}], ensure_ascii=False)
        elif "当前回答" in messages[-1]["content"]:
            out = "须提前 3 个工作日提交，经部门负责人审批[aaaaaaaaaa01]。金额超过 5000 元的，还须经财务部审批[aaaaaaaaaa02]。\n\n引用：aaaaaaaaaa01, aaaaaaaaaa02"
        else:
            out = "须提前 3 个工作日提交，经部门负责人审批[aaaaaaaaaa01]。\n\n引用：aaaaaaaaaa01"
        return {"text": out, "latency_ms": 1.0, "output_tokens": 10, "input_tokens": 100}


class NoOracleLeak(unittest.TestCase):
    @requires("experiments/p6_v5/data/frozen/hard_dev.jsonl", "experiments/p6_v5/data/frozen/test_v5.jsonl")
    def test_frozen_inputs_have_no_label_fields(self):
        for name in ("hard_dev", "test_v5"):
            for line in (V5 / "data" / "frozen" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.assertEqual(set(r), {"id", "question", "evidence", "input_sha256"})
                for e in r["evidence"]:
                    self.assertEqual(set(e), {"cid", "source", "doc", "text"})

    def test_rejects_evidence_with_label_fields(self):
        bad = [dict(EV[0], gold=True), EV[1]]
        with self.assertRaises(ValueError):
            pv2.run_planning_v2(FakeChat(), "问题", bad)

    def test_signature_only_question_and_evidence(self):
        ps = inspect.signature(pv2.run_planning_v2).parameters
        self.assertEqual([k for k, p in ps.items() if p.kind != p.KEYWORD_ONLY], ["chat", "question", "evidence"])
        self.assertEqual([k for k, p in ps.items() if p.kind == p.KEYWORD_ONLY], ["version"])

    def test_prompts_never_contain_labels(self):
        rec = {"id": "u", "question": "报销 8000 元要怎么申请？", "evidence": EV, "facts": [SENT_FACT], "fact_cids": ["aaaaaaaaaa02"],
               "gold_cids": ["aaaaaaaaaa02"], "reference_answer": SENT_REF, "required_facts": [SENT_FACT]}
        chat = FakeChat()
        frozen_like = {"question": rec["question"], "evidence": rec["evidence"]}       # run_eval_v5.gen 只把这两项传入
        r = pv2.run_planning_v2(chat, frozen_like["question"], frozen_like["evidence"])
        blob = "\n".join(chat.prompts)
        self.assertNotIn(SENT_FACT, blob)
        self.assertNotIn(SENT_REF, blob)
        for k in ("required_facts", "gold_cids", "reference_answer", "fact_cids"):
            self.assertNotIn(k, blob)
        self.assertTrue(r["check"]["revised"])        # 遗漏由规划器自己发现（第二个片段的事实），不是来自标注
        self.assertEqual(r["n_calls"], 4)              # 2 次逐片段抽取 + 生成 + 1 次修订

    def test_v21_prompts_never_contain_labels_and_rejects_refusal_revision(self):
        class RefuseOnRevise(FakeChat):
            def __call__(self, messages, n):
                r = super().__call__(messages, n)
                if "当前回答" in messages[-1]["content"]:
                    r["text"] = "资料中没有说明。\n\n引用：无"
                return r
        chat = RefuseOnRevise()
        r = pv2.run_planning_v2(chat, "报销 8000 元要怎么申请？", EV, version="2.1")
        blob = "\n".join(chat.prompts)
        self.assertNotIn(SENT_FACT, blob)
        self.assertNotIn(SENT_REF, blob)
        self.assertFalse(r["check"]["revised"])                 # 退化为拒答的修订必须被拒绝
        self.assertTrue(r["answer"].startswith("须提前 3 个工作日"))
        self.assertIsNotNone(r["check"]["stage2_answer"])
        self.assertTrue(all("raw" in c for c in r["calls"] if c["stage"] == "extract"))

    def test_gen_passes_only_question_and_evidence(self):
        src = inspect.getsource(run_eval_v5.cmd_gen)
        self.assertIn('run_planning_v2(chat, r["question"], [{k: e[k] for k in ("cid", "source", "doc", "text")} for e in r["evidence"]]', src)
        for k in FORBIDDEN_KEYS - {"answer"}:
            self.assertNotIn(f'r["{k}"]', src)

    def test_planning_module_reads_no_label_files(self):
        src = inspect.getsource(pv2)
        for s in ("hard_dev.jsonl", "built", "reference_answer\"]", "oracle", "required_facts\"]", "gold_cids"):
            self.assertNotIn(s, src)


if __name__ == "__main__":
    unittest.main()
