"""P6-v5 评分器 4.0.3（修订 4，2026-11-24 冻结）：非十六进制畸形引用 ID 的最小复现与回归测试。

* TestCurrent402Gap：用 scorer="4.0.2" 复现旧版本的漏判（最小复现保留，作为修订依据）。
* TestProposal403：规则模块（scorer_403_proposal 重新导出 scoring_v4 中的实现）的期望行为。
* TestOfficial403：正式评分器默认 4.0.3 已识别三类畸形 ID，且只影响引用判定。
"""
import unittest

from rag_finetuning.experiments.p6_v5.scripts import scorer_403_proposal as P
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4


def score_item_v4_402(rec, answer):
    return score_item_v4(rec, answer, scorer="4.0.2")

A, B = "aaaaaaaaaa01", "bbbbbbbbbb02"
REC = {"id": "t", "question": "名片要提前几个工作日申请？", "behavior": "answer", "facts": ["3个工作日"], "fact_cids": [A],
       "gold_cids": [A], "evidence": [{"cid": A, "source": "s", "text": "名片管理 > 申请\n名片须提前 3 个工作日申请。"},
                                     {"cid": B, "source": "s", "text": "名片管理 > 费用\n名片每盒 50 元。"}]}


def ans(cite):
    return f"名片须提前 3 个工作日申请[{cite}]。\n\n引用：{cite}"


class TestCurrent402Gap(unittest.TestCase):
    def test_version(self):
        self.assertEqual(SCORER_VERSION, "4.0.3")

    def test_valid_id_baseline(self):
        s = score_item_v4_402(REC, ans(A))
        self.assertTrue(s["citation_valid"])
        self.assertTrue(s["content_ok_v4"])

    def test_non_hex_malformed_missed(self):
        # 最小复现：d3a46887aml 不是合法 ID，4.0.2 没有识别
        s = score_item_v4_402(REC, ans("d3a46887aml"))
        self.assertEqual(s["malformed"], [])
        self.assertTrue(s["citation_valid"])

    def test_uppercase_ignored(self):
        s = score_item_v4_402(REC, ans(A.upper()))
        self.assertEqual(s["malformed"], [])
        self.assertEqual(s["fabricated"], [])

    def test_embedded_hex12_extracted(self):
        # 12 位十六进制夹在更长的串中：4.0.2 截出 aaaaaaaaaa01 当作有效引用
        s = score_item_v4_402(REC, ans("15c4744L" + A))
        self.assertTrue(s["citation_valid"])

    def test_content_not_affected_by_citation(self):
        self.assertTrue(score_item_v4_402(REC, ans("d3a46887aml"))["content_ok_v4"])


class TestProposal403(unittest.TestCase):
    def kinds(self, cite):
        return [k for _, k in P.malformed_403(ans(cite))]

    def test_non_hex_detected(self):
        for t in ("d3a46887aml", "72e16b3La4d0f5", "d979ml"):
            self.assertIn("P1_non_hex_chars", self.kinds(t), t)

    def test_uppercase_detected(self):
        self.assertIn("P2_uppercase_hex12", self.kinds(A.upper()))

    def test_embedded_detected(self):
        self.assertIn("P3_hex12_embedded", self.kinds("15c4744L" + A))

    def test_valid_and_internal_not_flagged(self):
        self.assertEqual(P.malformed_403(ans(A)), [])
        self.assertEqual(P.malformed_403("名片须提前 3 个工作日申请[F1]。\n\n引用：" + A), [])

    def test_plain_words_not_flagged(self):
        # 引用区域中的普通英文 / 系统名不应被当作 ID
        self.assertEqual(P.malformed_403("见[OA系统]、[HRMS]。\n\n引用：" + A), [])

    def test_wrong_length_hex_reported_as_already_handled(self):
        self.assertEqual(set(self.kinds("aaaaaaaaa02")), {"hex_wrong_length（4.0.2 已识别）"})  # 方括号与引用行各出现一次

    def test_proposed_score_changes_citation_only(self):
        s = P.score_item_403_proposed(REC, ans("d3a46887aml"))
        self.assertFalse(s["citation_valid_403"])
        self.assertFalse(s["format_ok_403"])
        self.assertTrue(s["content_ok_v4"])
        self.assertTrue(P.score_item_403_proposed(REC, ans(A))["citation_valid_403"])


class TestOfficial403(unittest.TestCase):
    def test_default_detects_all_three(self):
        for cite in ("d3a46887aml", A.upper(), "15c4744L" + A):
            s = score_item_v4(REC, ans(cite))
            self.assertFalse(s["citation_valid"], cite)
            self.assertFalse(s["format_ok"], cite)
            self.assertTrue(s["content_ok_v4"], cite)
            self.assertEqual(s["scorer_v4"], "4.0.3")

    def test_valid_unchanged(self):
        s = score_item_v4(REC, ans(A))
        self.assertTrue(s["citation_valid"])
        self.assertEqual(s["malformed"], [])

    def test_old_versions_reproducible(self):
        self.assertTrue(score_item_v4(REC, ans("d3a46887aml"), scorer="4.0.2")["citation_valid"])
        self.assertEqual(score_item_v4(REC, ans(A), scorer="4.0.2")["scorer_v4"], "4.0.2")


if __name__ == "__main__":
    unittest.main()
