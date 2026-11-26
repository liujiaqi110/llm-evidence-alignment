"""P6 software tests: dataset invariants (splits, leakage, provenance) and rule-based scorer behaviour.

No model, GPU or network needed:  python -m unittest discover -s rag_finetuning/tests -t . -v
"""

import json
import unittest

from rag_finetuning.common import BUILT, CANONICAL_REFUSAL, SPLITS, load_corpus, messages_for, read_jsonl
from rag_finetuning.evaluation.scoring import (classify, fact_hits, format_check, mcnemar_exact, normalize, paired,
                                               score_item, wilson)
from rag_finetuning.scripts import build_dataset
from rag_finetuning.compat.rag_pipeline import SYSTEM_PROMPT
from rag_finetuning.tests._private_data import requires


def _rec(behavior="answer", gold=("aaaaaaaaaaaa",), facts=("400",), forbid=(), fmt=None):
    return {"id": "x", "type": "A", "subtype": "", "behavior": behavior, "question": "杭州住宿每晚上限？",
            "evidence": [{"cid": "aaaaaaaaaaaa", "source": "a.md", "doc": "a", "section": 1, "text": "其他城市每晚 400 元。"},
                         {"cid": "bbbbbbbbbbbb", "source": "b.md", "doc": "b", "section": 1, "text": "无关内容 80 元。"}],
            "gold_cids": list(gold), "reference_answer": "", "facts": list(facts), "forbid": list(forbid), "format": fmt}


@requires("data/built/train.jsonl", "data/corpus")
class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.built = {s: read_jsonl(BUILT / f"{s}.jsonl") for s in SPLITS}
        cls.chunks = load_corpus()

    def test_build_is_reproducible_and_clean(self):
        """Re-running the builder (check mode) yields no problems and identical records."""
        self.assertEqual(build_dataset.main(["--check"]), 0)

    def test_splits_are_document_disjoint(self):
        docs = {s: {d for r in rows for d in r["source_docs"]} for s, rows in self.built.items()}
        self.assertFalse(docs["train"] & docs["test"])
        self.assertFalse(docs["train"] & docs["val"])
        self.assertFalse(docs["val"] & docs["test"])

    def test_no_test_answer_in_training_targets(self):
        targets = {r["reference_answer"] for r in self.built["train"]}
        for r in self.built["test"]:
            if r["behavior"] != "refuse":
                self.assertNotIn(r["reference_answer"], targets)

    def test_every_record_has_provenance(self):
        for rows in self.built.values():
            for r in rows:
                self.assertIn(r["generation_method"], {"claude_authored", "derived_gold_removed"})
                self.assertIn(r["review_status"], {"provisional", "reviewed"})
                self.assertTrue(r["dataset_version"])
        self.assertTrue(all(r["review_status"] == "provisional" for r in self.built["test"]),
                        "no test item may be marked reviewed by the builder itself")

    def test_reference_citations_exist_in_evidence(self):
        for rows in self.built.values():
            for r in rows:
                s = score_item(r, r["reference_answer"])
                self.assertEqual(s["fabricated_citations"], [], r["id"])
                self.assertTrue(s["pass"], (r["id"], s["errors"]))

    def test_all_five_capabilities_present_in_every_split_used_for_claims(self):
        for s in ("train", "test"):
            types = {r["type"] for r in self.built[s]}
            self.assertEqual(types, set("ABCDE"), s)
            beh = {r["behavior"] for r in self.built[s]}
            self.assertEqual(beh, {"answer", "partial", "conflict", "refuse"}, s)

    def test_prompt_is_the_platform_prompt(self):
        m = messages_for(self.built["test"][0])
        self.assertEqual(m[0]["content"], SYSTEM_PROMPT)
        self.assertIn("<资料 编号=", m[1]["content"])

    def test_derived_refusals_drop_gold(self):
        by_id = {r["id"]: r for rows in self.built.values() for r in rows}
        for r in by_id.values():
            if r["generation_method"] == "derived_gold_removed":
                src = by_id[r["derived_from"]]
                self.assertFalse(set(src["gold_cids"]) & {e["cid"] for e in r["evidence"]})
                self.assertEqual(r["reference_answer"], CANONICAL_REFUSAL)

    def test_chunk_ids_are_platform_format(self):
        for c in self.chunks.values():
            self.assertRegex(c.cid, r"^[0-9a-f]{12}$")


class TestScoring(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("1,800 元 １０：００"), "1800元10:00")

    def test_fact_alternatives(self):
        self.assertEqual(fact_hits("不予报销", ["不能|不予报销", "抬头"]), [True, False])

    def test_correct_answer_passes(self):
        s = score_item(_rec(), "按其他城市标准，每晚 400 元[aaaaaaaaaaaa]。\n\n引用：aaaaaaaaaaaa")
        self.assertTrue(s["pass"], s["errors"])
        self.assertEqual(s["citation_precision"], 1.0)

    def test_fabricated_citation_is_caught(self):
        s = score_item(_rec(), "每晚 400 元[cccccccccccc]。\n\n引用：cccccccccccc")
        self.assertIn("fabricated_citation", s["errors"])
        self.assertFalse(s["pass"])

    def test_unsupported_number_is_unfaithful(self):
        s = score_item(_rec(), "每晚 400 元，旺季 500 元[aaaaaaaaaaaa]。\n\n引用：aaaaaaaaaaaa")
        self.assertEqual(s["unsupported_numbers"], ["500"])
        self.assertFalse(s["faithful_rule"])

    def test_refusal_and_false_refusal(self):
        r = _rec(behavior="refuse", gold=(), facts=())
        self.assertTrue(score_item(r, CANONICAL_REFUSAL)["pass"])
        s = score_item(_rec(), CANONICAL_REFUSAL)
        self.assertEqual(s["pred_behavior"], "refuse")
        self.assertIn("false_refusal", s["errors"])

    def test_refusal_with_hallucinated_fact_fails(self):
        r = _rec(behavior="refuse", gold=(), facts=(), forbid=("98",))
        s = score_item(r, "资料没有提到，但一般是 98 天。\n\n引用：无")
        self.assertFalse(s["behavior_ok"])

    def test_conflict_needs_both_sources(self):
        r = _rec(behavior="conflict", gold=("aaaaaaaaaaaa", "bbbbbbbbbbbb"), facts=("400", "80"))
        ok = "资料之间存在冲突：400 元[aaaaaaaaaaaa]；80 元[bbbbbbbbbbbb]。\n\n引用：aaaaaaaaaaaa, bbbbbbbbbbbb"
        self.assertTrue(score_item(r, ok)["pass"])
        one = "每晚 400 元[aaaaaaaaaaaa]。\n\n引用：aaaaaaaaaaaa"
        self.assertIn("conflict_not_flagged", score_item(r, one)["errors"])

    def test_classify(self):
        self.assertEqual(classify("资料中没有说明。\n引用：无", []), "refuse")
        self.assertEqual(classify("A[x]。资料中没有说明 B。", ["x"]), "partial")
        self.assertEqual(classify("两份资料说法不一致", []), "conflict")
        self.assertEqual(classify("每晚 400 元", ["x"]), "answer")

    def test_formats(self):
        self.assertTrue(format_check("1. a\n2. b\n3. c", {"kind": "numbered_list", "min": 3}))
        self.assertFalse(format_check("a，b，c", {"kind": "numbered_list", "min": 3}))
        self.assertTrue(format_check("| a | b |\n|---|---|\n| 1 | 2 |", {"kind": "table"}))
        self.assertTrue(format_check("定义：x\n计算方式：y\n注意事项：z", {"kind": "sections", "headers": ["定义", "计算方式", "注意事项"]}))
        self.assertFalse(format_check("计算方式：y\n定义：x\n注意事项：z", {"kind": "sections", "headers": ["定义", "计算方式", "注意事项"]}))

    def test_stats(self):
        self.assertEqual(wilson(0, 0), None)
        lo, hi = wilson(5, 10)
        self.assertTrue(0.2 < lo < 0.5 < hi < 0.8)
        self.assertEqual(mcnemar_exact(0, 0), None)
        self.assertAlmostEqual(mcnemar_exact(5, 0), 0.0625)
        p = paired([False, True, True, False], [True, True, False, False], iters=200)
        self.assertEqual((p["improved"], p["regressed"], p["unchanged"]), (1, 1, 2))


if __name__ == "__main__":
    unittest.main()
