"""P6-v2 软件测试：Prompt 变体不变量、冻结输入完整性、盲测集独立性（无模型、无 GPU）。"""

import hashlib
import json
import unittest
from pathlib import Path

from rag_finetuning.tests._private_data import require_module

require_module("data/built/train.jsonl", "experiments/p6_v2/frozen_inputs/diagnostic_47.jsonl")

from rag_finetuning.common import BUILT, read_jsonl  # noqa: E402
from rag_finetuning.experiments.p6_v2.prompts import DEMOS, demo_docs, messages, prompt_fingerprint
from rag_finetuning.compat.rag_pipeline import SYSTEM_PROMPT

P = Path(__file__).resolve().parent.parent / "experiments" / "p6_v2"


class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.rec = read_jsonl(P / "frozen_inputs" / "diagnostic_47.jsonl")[0]

    def test_zero_shot_is_platform_prompt(self):
        m = messages(self.rec, "zero_shot")
        self.assertEqual(len(m), 2)
        self.assertEqual(m[0]["content"], SYSTEM_PROMPT)

    def test_few_shot_only_adds_examples(self):
        z, f = messages(self.rec, "zero_shot"), messages(self.rec, "few_shot")
        self.assertEqual(f[0], z[0])            # 相同系统指令
        self.assertEqual(f[-1], z[-1])          # 相同的真实问题与证据
        self.assertEqual(len(f), 2 + 2 * len(DEMOS))
        self.assertEqual([x["role"] for x in f[1:-1]], ["user", "assistant"] * len(DEMOS))

    def test_demos_come_from_train_only(self):
        train_docs = {e["doc"] for r in read_jsonl(BUILT / "train.jsonl") for e in r["evidence"]}
        other = {e["doc"] for s in ("val", "test") for r in read_jsonl(BUILT / f"{s}.jsonl") for e in r["evidence"]}
        self.assertTrue(demo_docs() <= train_docs)
        self.assertFalse(demo_docs() & other)
        self.assertEqual(len(DEMOS), 6)

    def test_prompt_frozen_before_blind(self):
        freeze = json.loads((P / "prompts" / "prompt_freeze.json").read_text(encoding="utf-8"))
        self.assertEqual(freeze["fewshot_demos_sha256"], prompt_fingerprint()["fewshot_demos_sha256"])


class TestFrozenInputs(unittest.TestCase):
    def test_hashes_and_no_reference_answers(self):
        for name in ("diagnostic_47", "blind_v1"):
            path = P / "frozen_inputs" / f"{name}.jsonl"
            man = json.loads((P / "frozen_inputs" / f"{name}.manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), man["frozen_file_sha256"], name)
            for r in read_jsonl(path):
                self.assertNotIn("reference_answer", r)
                self.assertEqual(man["items"][r["id"]], r["input_sha256"])

    def test_blind_is_independent(self):
        blind_docs = {e["doc"] for r in read_jsonl(P / "frozen_inputs" / "blind_v1.jsonl") for e in r["evidence"]}
        prev = {e["doc"] for s in ("train", "val", "test") for r in read_jsonl(BUILT / f"{s}.jsonl") for e in r["evidence"]}
        self.assertFalse(blind_docs & prev)
        self.assertFalse(blind_docs & demo_docs())
        n_unique = len({r["question"] for r in read_jsonl(P / "frozen_inputs" / "blind_v1.jsonl")})
        self.assertGreaterEqual(n_unique, 50)


if __name__ == "__main__":
    unittest.main()
