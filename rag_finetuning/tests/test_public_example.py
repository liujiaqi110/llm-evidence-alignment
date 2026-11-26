"""Public example + scorer smoke tests: run on a clean checkout with only PyYAML installed (no GPU, no private data)."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from examples.minimal import run_example
from rag_finetuning.experiments.p6_v5.scripts.scoring_v4 import SCORER_VERSION, score_item_v4

EX = Path(__file__).resolve().parents[2] / "examples" / "minimal"


class TestPublicExample(unittest.TestCase):
    def test_example_output_is_reproducible(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(run_example.main([]), 0)
        got = json.loads(buf.getvalue())
        want = json.loads((EX / "expected_output.json").read_text(encoding="utf-8"))
        self.assertEqual(got, want)

    def test_scorer_version_is_the_frozen_one(self):
        self.assertEqual(SCORER_VERSION, "4.0.3")

    def test_refusal_and_missing_fact(self):
        items = [json.loads(l) for l in (EX / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {r["id"]: r for r in items}
        refuse = score_item_v4(by_id["ex_04"], "根据提供的资料无法回答该问题。\n\n引用：无")
        self.assertTrue(refuse["content_ok_v4"])
        partial = score_item_v4(by_id["ex_02"], "须提前 48 小时在前台登记预约[48246596bc92]。\n\n引用：48246596bc92")
        self.assertFalse(partial["multi_complete_v5"])
        self.assertIn("30分钟", partial["missing_facts"])

    def test_example_data_is_regenerable(self):
        from examples.minimal import make_items
        lines = [json.dumps(r, ensure_ascii=False) for r in make_items.ITEMS]
        self.assertEqual(lines, (EX / "items.jsonl").read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    unittest.main()
