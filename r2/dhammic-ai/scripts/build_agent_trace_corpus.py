#!/usr/bin/env python3
"""Build a sanitized local agent-trace corpus for Dhammic-AI pretraining.

Sources are local Hermes / Claude / Codex / OpenCode traces and markdown notes.
The output intentionally contains redacted text only; known credential/value
patterns are removed before writing parquet shards.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_OUT = Path.home() / ".cache" / "dhammic_agent_traces" / "data"
VAL_FILENAME = "shard_06542.parquet"
TRAIN_FILENAME = "shard_00000.parquet"

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"hf_[A-Za-z0-9_\-]{20,}"), "<HF_TOKEN>"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "<OPENAI_KEY>"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{20,}"), "<SLACK_TOKEN>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "<GITHUB_TOKEN>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<AWS_ACCESS_KEY>"),
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|pwd|bearer)\s*[:=]\s*['\"]?[^'\"\s]{8,}"), r"\1=<REDACTED>"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]{12,}"), r"\1<REDACTED>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "<PRIVATE_KEY_BLOCK>"),
    # Long high-entropy blobs that are likely tokens/keys. Keep hashes/SHAs shorter than this.
    (re.compile(r"\b[A-Za-z0-9_\-+/=]{96,}\b"), "<LONG_SECRET_LIKE_BLOB>"),
]
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WS_RE = re.compile(r"[ \t]{3,}")

SKIP_PARTS = {
    ".cache", "node_modules", ".git", "venv", ".venv", "site-packages",
    "__pycache__", "plugins", "marketplaces", "statsig", "telemetry",
    "chrome", "cache", "uploads", "paste-cache",
}


def sanitize(text: str, max_chars: int = 12_000) -> str:
    text = CONTROL_RE.sub(" ", text)
    for pat, repl in SECRET_PATTERNS:
        text = pat.sub(repl, text)
    text = WS_RE.sub("  ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    if len(text) > max_chars:
        text = text[:max_chars] + "\n<TRUNCATED>"
    return text.strip()


def stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_PARTS:
        return True
    name = path.name.lower()
    if name.endswith((".env", ".pem", ".key", ".crt")):
        return True
    if any(x in name for x in ("credential", "token", "secret")) and path.suffix.lower() != ".md":
        return True
    try:
        if path.stat().st_size > 128 * 1024 * 1024:
            return True
    except OSError:
        return True
    return False


def json_to_text(obj) -> str:
    if isinstance(obj, dict):
        # Common chat/session shapes: prefer human-semantic fields, not raw full JSON.
        parts = []
        for key in ("role", "type", "source", "cwd", "model", "timestamp", "created_at"):
            if key in obj and isinstance(obj[key], (str, int, float, bool)):
                parts.append(f"{key}: {obj[key]}")
        for key in ("message", "content", "text", "prompt", "response", "summary", "title"):
            val = obj.get(key)
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, list):
                for item in val[:20]:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(json_to_text(item))
            elif isinstance(val, dict):
                parts.append(json_to_text(val))
        if parts:
            return "\n".join(p for p in parts if p)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)[:20_000]


def iter_jsonl_file(path: Path, source: str, max_docs_per_file: int) -> Iterable[dict]:
    yielded = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            chunk_parts: list[str] = []
            for line in f:
                if yielded >= max_docs_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = json_to_text(obj)
                except Exception:
                    txt = line
                txt = sanitize(txt)
                if not txt:
                    continue
                chunk_parts.append(txt)
                # pack nearby messages together to preserve event sequence structure
                if len("\n".join(chunk_parts)) >= 4000 or len(chunk_parts) >= 12:
                    body = "\n---\n".join(chunk_parts)
                    yield {"text": f"<source:{source}>\n<path:{path}>\n{body}"}
                    yielded += 1
                    chunk_parts.clear()
            if chunk_parts and yielded < max_docs_per_file:
                body = "\n---\n".join(chunk_parts)
                yield {"text": f"<source:{source}>\n<path:{path}>\n{body}"}
    except OSError:
        return


def iter_text_file(path: Path, source: str) -> Iterable[dict]:
    try:
        txt = sanitize(path.read_text(encoding="utf-8", errors="ignore"), max_chars=20_000)
    except OSError:
        return
    if len(txt) < 80:
        return
    # Split large markdown into chunks on headings/paragraph boundaries.
    start = 0
    step = 12_000
    while start < len(txt):
        chunk = txt[start:start + step]
        if len(chunk) >= 80:
            yield {"text": f"<source:{source}>\n<path:{path}>\n{chunk}"}
        start += step


def discover_files(include_markdown: bool) -> list[tuple[Path, str]]:
    specs = [
        ("~/.hermes/sessions/**/*.jsonl", "hermes_session"),
        ("~/.hermes/cron/output/**/*.jsonl", "hermes_cron"),
        ("~/.claude/projects/**/*.jsonl", "claude_project"),
        ("~/.claude/transcripts/**/*.jsonl", "claude_transcript"),
        ("~/.codex/sessions/**/*.jsonl", "codex_session"),
    ]
    if include_markdown:
        specs += [
            ("~/.hermes/research/**/*.md", "hermes_research_md"),
            ("~/.hermes/plans/**/*.md", "hermes_plan_md"),
            ("~/.claude/projects/**/*.md", "claude_project_md"),
            ("~/.codex/**/*.md", "codex_md"),
            ("~/.config/opencode/**/*.md", "opencode_md"),
        ]
    out: list[tuple[Path, str]] = []
    for pat, source in specs:
        for f in glob.glob(os.path.expanduser(pat), recursive=True):
            p = Path(f)
            if p.is_file() and not should_skip(p):
                out.append((p, source))
    out.sort(key=lambda ps: ps[0].stat().st_mtime if ps[0].exists() else 0, reverse=True)
    return out


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=pa.schema([("text", pa.string())]))
    pq.write_table(table, path, compression="zstd", row_group_size=256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-docs", type=int, default=80_000)
    ap.add_argument("--max-docs-per-file", type=int, default=64)
    ap.add_argument("--max-chars", type=int, default=300_000_000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--include-markdown", action="store_true")
    args = ap.parse_args()

    docs: list[dict] = []
    total_chars = 0
    files = discover_files(args.include_markdown)
    seen = set()
    by_source: dict[str, int] = {}
    for path, source in files:
        iterator = iter_jsonl_file(path, source, args.max_docs_per_file) if path.suffix == ".jsonl" else iter_text_file(path, source)
        for row in iterator:
            txt = row["text"]
            sid = stable_id(txt)
            if sid in seen:
                continue
            seen.add(sid)
            docs.append(row)
            total_chars += len(txt)
            by_source[source] = by_source.get(source, 0) + 1
            if len(docs) >= args.max_docs or total_chars >= args.max_chars:
                break
        if len(docs) >= args.max_docs or total_chars >= args.max_chars:
            break

    if len(docs) < 100:
        raise SystemExit(f"too few docs collected: {len(docs)}")

    # Stable deterministic val split by content hash, not source order.
    docs.sort(key=lambda r: stable_id(r["text"]))
    n_val = max(64, int(len(docs) * args.val_frac))
    val = docs[:n_val]
    train = docs[n_val:]
    write_parquet(train, args.out / TRAIN_FILENAME)
    write_parquet(val, args.out / VAL_FILENAME)
    manifest = {
        "train_docs": len(train),
        "val_docs": len(val),
        "total_docs": len(docs),
        "total_chars": total_chars,
        "sources": by_source,
        "redaction": "tokens, api keys, bearer auth, private-key blocks, long secret-like blobs",
        "train_file": str(args.out / TRAIN_FILENAME),
        "val_file": str(args.out / VAL_FILENAME),
    }
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
