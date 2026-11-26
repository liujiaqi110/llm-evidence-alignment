"""Release guard: fail if the repository contains anything outside the authorised public scope.

Checks every file Git tracks (or, outside a Git checkout, every file under the repository root):
weights / checkpoints, unpublished experiment data (test_v5 annotations, corpora, preference pairs, per-item model
outputs), local absolute paths, personal e-mail addresses, credential-like strings and oversized files.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE = 5 * 1024 * 1024
MAX_TOTAL = 50 * 1024 * 1024

FORBIDDEN_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".onnx", ".h5", ".npz", ".pkl",
                 ".sqlite", ".db", ".parquet", ".arrow")
FORBIDDEN_NAME = re.compile(r"(^|/)(adapter_model[^/]*|optimizer\.pt|\.env|.*\.pem|id_rsa[^/]*)$")
# data and per-item outputs are not published; only the allow-listed text formats below may appear
ALLOWED_DATA = re.compile(
    r"^(examples/minimal/[^/]+\.jsonl"
    r"|rag_finetuning/experiments/p6_v[45]/results/.+(train_log|step_stats)[^/]*\.(jsonl|csv)"
    r"|rag_finetuning/experiments/p6_v5/final/FINAL_METRICS_TABLE\.csv)$")
DATA_EXT = (".jsonl", ".csv", ".tsv", ".html", ".yaml.bak")
FORBIDDEN_PATH = re.compile(
    r"(/data/test_v5/|/frozen/test_v5\.jsonl$|/results/test_v5/|per_item_test_v5|/final/failure_data\.json$"
    r"|/data/train_pref/|/data/train_v5/|/data/corpus/|/data/built/|/data/drafts/|/blind/|/frozen_inputs/[^/]+\.jsonl$"
    r"|HUMAN_REVIEW_DASHBOARD|PREFERENCE_PAIR_AUDIT|/generations[^/]*$)")
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\/]{1,2}(?:AI|Ai|ai|Users|Documents)\b|/home/[a-z]|/Users/[A-Za-z]")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com|anthropic\.com|example\.(?:com|org))"
                   r"[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}")
SECRET = re.compile(r"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}"
                    r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})")
SELF = {"scripts/check_release.py"}


def tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
        files = [f for f in out.decode("utf-8").split("\0") if f]
        if files:
            return files
    except (OSError, subprocess.CalledProcessError):
        pass
    return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]


def main() -> int:
    problems: list[str] = []
    files = tracked()
    total = 0
    for f in files:
        p = ROOT / f
        if not p.exists():
            continue
        size = p.stat().st_size
        total += size
        low = f.lower()
        if low.endswith(FORBIDDEN_EXT) or FORBIDDEN_NAME.search(low):
            problems.append(f"weights/secret file type: {f}")
        if FORBIDDEN_PATH.search(f):
            problems.append(f"unpublished data path: {f}")
        if low.endswith(DATA_EXT) and not ALLOWED_DATA.match(f):
            problems.append(f"per-item data / output file not in allow-list: {f}")
        if size > MAX_FILE:
            problems.append(f"file larger than {MAX_FILE // 2**20} MiB: {f}")
        if f in SELF:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for rx, what in ((ABS_PATH, "local absolute path"), (EMAIL, "e-mail address"), (SECRET, "credential-like string")):
            m = rx.search(text)
            if m:
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{what} in {f}:{line}: {m.group(0)[:40]}")
    if total > MAX_TOTAL:
        problems.append(f"repository larger than {MAX_TOTAL // 2**20} MiB ({total / 2**20:.1f} MiB)")
    for pr in problems:
        print("FAIL", pr)
    print(f"checked {len(files)} files, {total / 2**20:.2f} MiB: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
