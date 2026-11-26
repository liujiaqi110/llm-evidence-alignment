"""Helpers for tests that need the unpublished experiment data.

The public repository does not ship the experiment datasets (corpora, built splits, frozen dev/test sets,
preference pairs). Tests that check invariants of those files are skipped when the files are absent and run
unchanged when the private data is placed at the original paths. See docs/DATA_AVAILABILITY.md.
"""
import unittest
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REASON = "unpublished experiment data not present (see docs/DATA_AVAILABILITY.md)"


def requires(*rel_paths):
    """Skip the decorated test / class unless every path (relative to rag_finetuning/) exists."""
    ok = all((PKG / p).exists() for p in rel_paths)
    return unittest.skipUnless(ok, f"{REASON}: {', '.join(rel_paths)}")


def require_module(*rel_paths):
    """Raise SkipTest at import time (for test modules whose imports read private data)."""
    if not all((PKG / p).exists() for p in rel_paths):
        raise unittest.SkipTest(f"{REASON}: {', '.join(rel_paths)}")
