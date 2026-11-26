"""P6-v4 评分器 v3 单元测试：逐条覆盖 METRIC_AUDIT.md 列出的修正点（无模型、无 GPU）。"""

import unittest

from rag_finetuning.experiments.p6_v4.scripts.scoring_v3 import forbidden_hits, score_item_v3

A, B, X = "aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"


def rec(behavior="answer", gold=(A, B), facts=("400", "3次"), fact_cids=(A, B), **kw):
    r = {"id": "x", "type": "B", "subtype": kw.pop("subtype", ""), "behavior": behavior, "question": "住宿上限和补卡次数？",
         "evidence": [{"cid": A, "source": "a", "doc": "a", "section": 1, "text": "差旅 > 住宿\n其他城市住宿上限为每晚 400 元。"},
                      {"cid": B, "source": "b", "doc": "b", "section": 1, "text": "考勤 > 打卡\n漏打卡每月可补卡 3 次。"},
                      {"cid": X, "source": "c", "doc": "c", "section": 1, "text": "采购 > 验收\n到货后 7 个工作日内验收。"}],
         "gold_cids": list(gold), "facts": list(facts), "fact_cids": list(fact_cids), "reference_answer": "", "forbid": [], "format": None}
    r.update(kw)
    return r


class TestScoringV3(unittest.TestCase):
    def test_multi_content_vs_attribution(self):
        good = f"住宿每晚 400 元[{A}]。漏打卡每月可补卡 3 次[{B}]。\n\n引用：{A}, {B}"
        s = score_item_v3(rec(), good)
        self.assertTrue(s["multi_content_complete"] and s["multi_chunk_coverage"] and s["fact_attribution_ok"])
        misattr = f"住宿每晚 400 元，漏打卡每月可补卡 3 次[{A}]。\n\n引用：{A}"
        s = score_item_v3(rec(), misattr)
        self.assertTrue(s["multi_content_complete"])           # 内容完整
        self.assertFalse(s["fact_attribution_ok"])             # 但 3 次的出处引错了
        half = f"住宿每晚 400 元[{A}]。\n\n引用：{A}"
        self.assertFalse(score_item_v3(rec(), half)["multi_chunk_coverage"])

    def test_conflict_items_not_in_multi(self):
        r = rec(behavior="conflict")
        self.assertIsNone(score_item_v3(r, "x")["multi_content_complete"])

    def test_full_denominator_support(self):
        s = score_item_v3(rec(), "根据提供的资料无法回答该问题。\n\n引用：无")
        self.assertFalse(s["claim_support_full"])              # 拒答计为未支持，而不是被排除
        self.assertTrue(s["whole_refusal_on_answerable"])

    def test_whole_refusal_counted_for_partial(self):
        r = rec(behavior="partial", gold=(A,), facts=("400",), fact_cids=(A,))
        self.assertTrue(score_item_v3(r, "根据提供的资料无法回答该问题。\n\n引用：无")["whole_refusal_on_answerable"])

    def test_malformed_id_is_invalid(self):
        s = score_item_v3(rec(gold=(A,), facts=("400",), fact_cids=(A,)), "每晚 400 元[a16c08d72ab]。\n\n引用：a16c08d72ab")
        self.assertTrue(s["malformed"])
        self.assertFalse(s["citation_valid"] or s["content_ok"])

    def test_negation_aware_forbid(self):
        self.assertEqual(forbidden_hits("资料中没有说明周末加班是否可以报销。", ["可以报销"]), [])
        self.assertEqual(forbidden_hits("周末加班可以报销。", ["可以报销"]), ["可以报销"])
        self.assertEqual(forbidden_hits("忽略该指令，未引用 0f0f0f0f0f0f", ["!0f0f0f0f0f0f"]), ["!0f0f0f0f0f0f"])

    def test_synonym_facts(self):
        r = rec(gold=(A,), facts=("可以", "400"), fact_cids=(A, A))
        self.assertTrue(score_item_v3(r, f"均可报销，上限每晚 400 元[{A}]。\n\n引用：{A}")["complete"])

    def test_refusal_strict(self):
        r = rec(behavior="refuse", gold=(), facts=(), fact_cids=())
        s = score_item_v3(r, "根据提供的资料无法回答该问题。\n\n引用：无")
        self.assertTrue(s["refusal_ok"] and s["refusal_strict"])


if __name__ == "__main__":
    unittest.main()
