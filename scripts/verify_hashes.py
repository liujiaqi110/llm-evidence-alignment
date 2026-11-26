"""Verify that the published copies of hash-frozen files are byte-identical to what the experiment records froze.

Each entry names the file, its frozen sha256 and the record that holds that hash. The script checks both that the
file hashes to the value and that the value really appears in the record, so the list cannot drift from the records.
Files listed here are marked ``-text`` in .gitattributes so Git never rewrites their line endings.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = "rag_finetuning/experiments"
FROZEN = [
    # (file, sha256, record that froze it)
    (f"{E}/p6_v5/scripts/scoring_v4.py",
     "a230bfe1a4f5e884ccfaeec3e32746da2c92ada630bb4b64ae47b88c5ad6fe59", f"{E}/p6_v5/configs/test_v5_final_freeze.json"),
    (f"{E}/p6_v5/data/frozen/test_v5.manifest.json",
     "58f0a41e503e531a15d986750052626f8db19485145de73d90d623b57292f3a1", f"{E}/p6_v5/configs/test_v5_final_freeze.json"),
    (f"{E}/p6_v4/scripts/scoring_v3.py", None, f"{E}/p6_v4/configs/freeze.json"),
    (f"{E}/p6_v4/scripts/eng.py", None, f"{E}/p6_v4/configs/freeze.json"),
    (f"{E}/p6_v3/scripts/scoring_v2.py", None, f"{E}/p6_v3/configs/freeze.json"),
]


def main() -> int:
    bad = 0
    for rel, want, record in FROZEN:
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        rec_text = (ROOT / record).read_text(encoding="utf-8")
        ok_rec = got in rec_text
        ok_val = want is None or got == want
        status = "OK " if ok_rec and ok_val else "BAD"
        bad += status == "BAD"
        print(f"{status} {got[:16]}…  {rel}  (recorded in {record}: {'yes' if ok_rec else 'NO'})")
    print("all frozen hashes verified" if not bad else f"{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
