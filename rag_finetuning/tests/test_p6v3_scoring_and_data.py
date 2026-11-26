"""P6-v3 软件测试（无模型 / GPU）：评分器 v2 的修正点、data_v3 不变量、test_v3 冻结完整性。"""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from rag_finetuning.common import read_jsonl
from rag_finetuning.experiments.p6_v3.scripts.scoring_v2 import (claims_of, conflict_claimed, paired_cluster,
                                                                score_item_v2)

V3 = Path(__file__).resolve().parent.parent / "experiments" / "p6_v3"
A, B, X = "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"


def rec(behavior="answer", gold=(A,), facts=("400",), **kw):
    r = {"id": "x", "type": "A", "subtype": kw.pop("subtype", ""), "behavior": behavior, "question": "杭州出差住宿每晚上限是多少？",
         "evidence": [{"cid": A, "source": "a.md", "doc": "a", "section": 1, "text": "差旅制度 > 住宿\n其他城市住宿上限为每晚 400 元。"},
                      {"cid": B, "source": "b.md", "doc": "b", "section": 1, "text": "通知 > 住宿\n其他城市住宿上限为每晚 500 元。"},
                      {"cid": X, "source": "c.md", "doc": "c", "section": 1, "text": "考勤 > 打卡\n漏打卡每月可补卡 3 次。"}],
         "gold_cids": list(gold), "reference_answer": "", "facts": list(facts), "forbid": [], "format": None}
    r.update(kw)
    return r


from rag_finetuning.tests._private_data import requires  # noqa: E402

class TestScoringV2(unittest.TestCase):
    def test_conflict_keyword_alone_is_not_enough(self):
        r = rec("conflict", gold=(A, B), facts=("400", "500"))
        only_word = "资料之间存在冲突，请以最新规定为准。\n\n引用：" + A
        self.assertFalse(score_item_v2(r, only_word)["behavior_ok"])
        good = f"资料之间存在冲突：制度规定每晚 400 元[{A}]；通知写的是每晚 500 元[{B}]。\n\n引用：{A}, {B}"
        self.assertTrue(score_item_v2(r, good)["behavior_ok"])

    def test_negated_conflict_is_not_a_conflict_claim(self):
        self.assertFalse(conflict_claimed("两者适用条件不同，并不矛盾。"))
        self.assertTrue(conflict_claimed("两份资料存在冲突。"))

    def test_no_conflict_item_claiming_conflict_fails(self):
        r = rec("answer", gold=(B,), facts=("500",), no_conflict=True)
        bad = f"资料之间存在冲突：现行为每晚 500 元[{B}]，旧规定为 400 元[{A}]。\n\n引用：{B}, {A}"
        ok = f"按现行通知，其他城市住宿上限为每晚 500 元[{B}]。\n\n引用：{B}"
        self.assertFalse(score_item_v2(r, bad)["behavior_ok"])
        self.assertTrue(score_item_v2(r, ok)["behavior_ok"])

    def test_refusal_phrase_with_fabricated_fact_is_wrong(self):
        r = rec("refuse", gold=(), facts=(), forbid=["98"])
        self.assertFalse(score_item_v2(r, "资料中没有说明，但一般产假是 98 天。\n\n引用：无")["behavior_ok"])
        s = score_item_v2(r, "根据提供的资料无法回答该问题。\n\n引用：无")
        self.assertTrue(s["behavior_ok"] and s["refusal_strict"])

    def test_valid_but_irrelevant_citation_is_unsupported(self):
        r = rec()
        s = score_item_v2(r, f"其他城市住宿上限为每晚 400 元[{X}]。\n\n引用：{X}")
        self.assertTrue(s["format_ok"])                      # 引用 ID 合法、格式正确
        self.assertFalse(s["all_claims_supported"])          # 但被引片段不支持该陈述
        s2 = score_item_v2(r, f"其他城市住宿上限为每晚 400 元[{A}]。\n\n引用：{A}")
        self.assertTrue(s2["all_claims_supported"])

    def test_conclusion_sentence_inherits_following_citation(self):
        cl = claims_of(f"不可以超过。其他城市住宿上限为每晚 400 元[{A}]。")
        self.assertTrue(all(c["cites"] == [A] for c in cl))

    def test_cluster_bootstrap(self):
        rows_a = [{"rec": rec(), "s": {"v": False}} for _ in range(4)]
        rows_b = [{"rec": rec(), "s": {"v": True}} for _ in range(4)]
        p = paired_cluster(rows_a, rows_b, "v", iters=200)
        self.assertEqual((p["diff"], p["n_clusters"]), (1.0, 1))


@requires("experiments/p6_v3/data/v3/full/train.jsonl", "experiments/p6_v3/data/test_v3/built/test_v3.jsonl")
class TestDataV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = read_jsonl(V3 / "data" / "v3" / "full" / "train.jsonl")
        cls.test = read_jsonl(V3 / "data" / "test_v3" / "built" / "test_v3.jsonl")

    def test_no_duplicate_questions_or_behavior_conflicts(self):
        q = Counter(r["question"] for r in self.full)
        self.assertEqual(max(q.values()), 1)
        self.assertFalse(any(r["generation_method"] == "derived_gold_removed" for r in self.full))

    def test_required_fields(self):
        need = {"sample_id", "source_document_ids", "question", "evidence", "answer", "required_facts", "evidence_spans",
                "citation_ids", "answerability", "task_category", "generation_method", "review_status"}
        for r in self.full:
            self.assertTrue(need <= set(r), r["id"])
            self.assertEqual(r["review_status"], "provisional")

    def test_refusal_evidence_lacks_answer_facts(self):
        for r in self.full:
            if r["behavior"] == "refuse" and r["id"].startswith("v3_ref"):
                ev = "".join(e["text"] for e in r["evidence"]).replace(" ", "")
                for f in r["forbid"]:
                    self.assertNotRegex(ev, f, r["id"])

    def test_docs_disjoint_from_test(self):
        a = {d for r in self.full for d in r["source_docs"]}
        b = {d for r in self.test for d in r["source_docs"]}
        self.assertFalse(a & b)

    def test_test_v3_frozen_hash(self):
        man = json.loads((V3 / "data" / "frozen" / "test_v3.manifest.json").read_text(encoding="utf-8"))
        p = V3 / "data" / "frozen" / "test_v3.jsonl"
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), man["frozen_sha256"])
        self.assertTrue(all("reference_answer" not in r for r in read_jsonl(p)))
        self.assertEqual(len(self.test), 81)

    def test_references_pass_scorer(self):
        for r in self.test + self.full:
            s = score_item_v2(r, r["reference_answer"])
            self.assertTrue(s["content_ok"] and s["format_ok"], r["id"])


if __name__ == "__main__":
    unittest.main()
