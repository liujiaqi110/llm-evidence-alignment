"""P6-v5 评分器 4.0.2 回归测试：区分内部事实编号、引用编号与真正的证据外数值。"""
import unittest

from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4

EV = [{"cid": "aaaaaaaaaa01", "source": "x.md", "doc": "x", "text": "制度 > 申请\n申请须提前 3 个工作日提交，经部门负责人审批。"},
      {"cid": "aaaaaaaaaa02", "source": "x.md", "doc": "x", "text": "制度 > 金额\n金额超过 5000 元的，还须经财务部审批。"}]
REC = {"id": "u1", "behavior": "answer", "question": "报销 8000 元要怎么申请？", "evidence": EV, "gold_cids": ["aaaaaaaaaa01", "aaaaaaaaaa02"],
       "facts": ["3个工作日", "部门负责人", "财务部"], "fact_cids": ["aaaaaaaaaa01", "aaaaaaaaaa01", "aaaaaaaaaa02"], "forbid": [],
       "reference_answer": "", "subtype": ""}
BODY = "须提前 3 个工作日提交，经部门负责人审批[aaaaaaaaaa01]。金额超过 5000 元，还须经财务部审批[aaaaaaaaaa02]。"
CITE = "\n\n引用：aaaaaaaaaa01, aaaaaaaaaa02"


class ScorerRegression402(unittest.TestCase):
    def test_version(self):
        # 修订 4（2026-11-24）后正式版本为 4.0.3；本文件的 4.0.2 内部编号规则在 4.0.3 中保持不变
        self.assertEqual(SCORER_VERSION, "4.0.3")

    def test_list_style_fact_ids_not_external_numbers(self):
        ans = "- F7：须提前 3 个工作日提交，经部门负责人审批[aaaaaaaaaa01]。\n- F1：金额超过 5000 元，还须经财务部审批[aaaaaaaaaa02]。" + CITE
        new, old = score_item_v4(REC, ans), score_item_v4(REC, ans, scorer="4.0.1")
        self.assertEqual(new["unsupported_numbers_v4"], [])
        self.assertTrue(new["content_ok_v4"])
        self.assertEqual(new["internal_fact_ids"], ["F7", "F1"])
        self.assertTrue(old["unsupported_numbers_v4"])           # 4.0.1 把 7、1 判为证据外数字（复现原缺陷）
        self.assertFalse(old["content_ok_v4"])

    def test_bracket_fact_ids_not_external_numbers(self):
        ans = BODY.replace("[aaaaaaaaaa02]", "[F2, F3]") + CITE
        s = score_item_v4(REC, ans)
        self.assertEqual(s["unsupported_numbers_v4"], [])
        self.assertEqual(s["internal_fact_ids"], ["F2", "F3"])

    def test_real_external_number_still_flagged(self):
        ans = BODY.replace("5000 元", "9999 元") + CITE
        s = score_item_v4(REC, ans)
        self.assertIn("9999", s["unsupported_numbers_v4"])
        self.assertFalse(s["content_ok_v4"])

    def test_external_number_next_to_fact_id_still_flagged(self):
        ans = "- F1：须提前 7 个工作日提交，经部门负责人审批[aaaaaaaaaa01]。金额超过 5000 元，还须经财务部审批[aaaaaaaaaa02]。" + CITE
        s = score_item_v4(REC, ans)
        self.assertIn("7", s["unsupported_numbers_v4"])        # 7 在“7 个工作日”中，是真正的证据外数值

    def test_citation_and_malformed_ids_not_numbers(self):
        s = score_item_v4(REC, BODY + CITE)
        self.assertEqual(s["unsupported_numbers_v4"], [])
        m = score_item_v4(REC, BODY.replace("[aaaaaaaaaa02]", "[aaaaaaaaa02]") + CITE)
        self.assertEqual(m["unsupported_numbers_v4"], [])
        self.assertFalse(m["citation_valid"])

    def test_words_containing_f_digits_not_stripped(self):
        ans = BODY.replace("金额超过 5000 元", "金额超过 5000 元（附件 PDF2）") + CITE
        s = score_item_v4(REC, ans)
        self.assertEqual(s["internal_fact_ids"], [])
        self.assertIn("2", s["unsupported_numbers_v4"])        # PDF2 中的 2 不是内部编号，仍按证据外数字计

    def test_question_numbers_allowed(self):
        ans = "8000 元超过 5000 元，还须经财务部审批[aaaaaaaaaa02]；须提前 3 个工作日提交，经部门负责人审批[aaaaaaaaaa01]。" + CITE
        self.assertEqual(score_item_v4(REC, ans)["unsupported_numbers_v4"], [])


if __name__ == "__main__":
    unittest.main()
