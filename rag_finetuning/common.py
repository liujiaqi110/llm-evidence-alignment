"""Shared, dependency-free helpers for the P6 fine-tuning experiment.

* Corpus → chunks: one chunk per ``## `` section, prefixed with ``<doc title> > <section title>`` (same idea as the
  platform's context-inheritance step); chunk IDs use the platform scheme ``md5(f"{source}:{idx}:{text[:50]}")[:12]``.
* Prompt: the EXACT platform prompt (``rag_finetuning.compat.rag_pipeline.SYSTEM_PROMPT`` / ``build_messages``, rag_v1.0), so a
  model trained here sees the same instructions and context markup as the production pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rag_finetuning.compat.rag_pipeline import (PROMPT_VERSION, REFUSAL_TEXT, SYSTEM_PROMPT, build_messages,  # noqa: F401
                                     prompt_hash)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CORPUS = ROOT / "data" / "corpus"
QA_DIR = ROOT / "data" / "qa"
BUILT = ROOT / "data" / "built"
SPLITS = ("train", "val", "test")
DATASET_VERSION = "p6-rag-sft-v1.0"

# Documents that are only ever used where a QA item names them explicitly (conflict sources, prompt-injection letters).
# They are never drawn as random distractors, so an "answer" item can never silently acquire a conflicting chunk.
EXPLICIT_ONLY = {"finance_faq", "admin_faq", "hr_portal_faq", "vendor_letter", "partner_letter"}

CANONICAL_REFUSAL = f"{REFUSAL_TEXT}\n\n引用：无"


@dataclass(frozen=True)
class Chunk:
    cid: str
    doc: str          # file stem
    source: str       # file name, shown to the model as 来源
    section: int      # 1-based section number
    title: str
    text: str
    split: str


def chunk_document(path: Path, split: str) -> List[Chunk]:
    raw = path.read_text(encoding="utf-8")
    doc_title = next((ln[2:].strip() for ln in raw.splitlines() if ln.startswith("# ")), path.stem)
    out, idx = [], 0
    for block in re.split(r"(?m)^## ", raw)[1:]:
        head, _, body = block.partition("\n")
        body = body.strip()
        text = f"{doc_title} > {head.strip()}\n{body}"
        cid = hashlib.md5(f"{path.name}:{idx}:{text[:50]}".encode()).hexdigest()[:12]
        idx += 1
        out.append(Chunk(cid, path.stem, path.name, idx, head.strip(), text, split))
    return out


def load_corpus() -> Dict[str, Chunk]:
    """``{"doc#section": Chunk}`` over all splits; raises if a document name occurs in more than one split."""
    chunks: Dict[str, Chunk] = {}
    seen: Dict[str, str] = {}
    for split in SPLITS:
        for p in sorted((CORPUS / split).glob("*.md")):
            if p.stem in seen:
                raise ValueError(f"document {p.stem} appears in both {seen[p.stem]} and {split}")
            seen[p.stem] = split
            for c in chunk_document(p, split):
                chunks[f"{c.doc}#{c.section}"] = c
    return chunks


def messages_for(record: dict) -> List[Dict[str, str]]:
    """Platform prompt for a built record (system + user). Evidence order is the record's (already shuffled) order."""
    parts = [(e["cid"], e["text"]) for e in record["evidence"]]
    sources = {e["cid"]: e["source"] for e in record["evidence"]}
    return build_messages(record["question"], parts, sources)


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
