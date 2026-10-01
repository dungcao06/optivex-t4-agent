"""Claims that Track 4 scorer 5.2.2 cannot mark false.

From scorer 5.2.0 a claim earns nothing and each false claim multiplies the unit's score by
1 - F / (F + min(T, 3E)). So every claim here is a short verbatim quote of a span its entity may
cite: a corpus document the manifest labels for that entity, a shared document, or the
entity's own row of the task table (doc_id "task"). Ownership comes from the read-only corpus
manifest; the scorer applies the same labels from the unit manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

TASK_DOC_ID = "task"
MAX_CLAIM_CHARS = 400   # Densest public 600-char window measured 375 judge tokens (cap 400).
MIN_CLAIM_CHARS = 20
MAX_CLAIMS = 3

Ownership = dict[str, tuple[frozenset[str], bool]]


def load_ownership(corpus_dir: Path) -> Ownership | None:
    """doc_id -> (entity_ids, shared) from corpus/manifest.json, or None if unreadable."""
    try:
        files = json.loads((Path(corpus_dir) / "manifest.json").read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(files, list):
        return None
    owners: Ownership = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        name = entry["path"].rsplit("/", 1)[-1]
        if not name.endswith(".json") or name == "manifest.json":
            continue
        ids = entry.get("entity_ids", [])
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            continue
        owners[name[: -len(".json")]] = (frozenset(ids), entry.get("shared") is True)
    return owners


def may_cite(owners: Ownership | None, doc_id: str, entity_id: str) -> bool:
    """The scorer's rule: shared, or labeled with this entity. Unknown ownership is not citable."""
    if not owners or doc_id not in owners:
        return False
    entity_ids, shared = owners[doc_id]
    return shared or entity_id in entity_ids


def trim_quote(text: str, limit: int = MAX_CLAIM_CHARS) -> str:
    """A verbatim prefix of at most `limit` characters, cut at a space when possible."""
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit + 1)
    return text[: cut if cut >= MIN_CLAIM_CHARS else limit].rstrip()


def task_row(task: dict, entity_id: str) -> tuple[str, int] | None:
    """The entity's line of the scorer's task table and its start offset."""
    offset = 0
    for row in task.get("entities") or []:
        line = json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
        if isinstance(row, dict) and row.get("entity_id") == entity_id:
            return line, offset
        offset += len(line) + 1
    return None


def task_row_claim(task: dict, entity_id: str) -> dict | None:
    """A verbatim quote of the entity's own task row: always citable by that entity."""
    found = task_row(task, entity_id)
    if found is None:
        return None
    line, start = found
    text = trim_quote(line)
    return {"doc_id": TASK_DOC_ID, "span_start": start, "span_end": start + len(text), "claim": text}


def build_claims(items: object, chunks: list, task: dict, entity_id: str,
                 owners: Ownership | None) -> list[dict]:
    """Verbatim, citable quotes from the model's evidence; the entity's task row if none survive."""
    claims: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for item in (items if isinstance(items, list) else [])[:8]:
        if len(claims) == MAX_CLAIMS:
            break
        if not isinstance(item, dict):
            continue
        doc_id, quote = item.get("doc_id"), item.get("quote")
        if not isinstance(doc_id, str) or not isinstance(quote, str):
            continue
        quote = trim_quote(quote.strip())
        if len(quote) < MIN_CLAIM_CHARS or not may_cite(owners, doc_id, entity_id):
            continue
        for chunk in chunks:
            start = chunk.text.find(quote) if chunk.doc_id == doc_id else -1
            if start >= 0:
                source = (doc_id, chunk.span_start + start, chunk.span_start + start + len(quote))
                if source not in seen:
                    seen.add(source)
                    claims.append({"doc_id": doc_id, "span_start": source[1],
                                   "span_end": source[2], "claim": quote})
                break
    if not claims:
        row = task_row_claim(task, entity_id)
        if row is not None:
            claims.append(row)
    return claims


def citable_note(owners: Ownership | None, entity_id: str, chunks: list) -> str:
    """Prompt suffix naming the excerpts this entity may quote."""
    ids = sorted({chunk.doc_id for chunk in chunks if may_cite(owners, chunk.doc_id, entity_id)})
    if not ids:
        return ("\nCITATION RULE: none of these excerpts may be quoted for this entity. Use them "
                'as context and return "evidence": [].')
    return ("\nCITATION RULE: quote only from doc_id " + ", ".join(ids) + ". Other excerpts are "
            "context about other entities: use them, never quote them. Each quote is one exact "
            "sentence or table row of at most 300 characters that contains a figure.")
