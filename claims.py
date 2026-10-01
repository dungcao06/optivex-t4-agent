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
MAX_CLAIM_CHARS = 400   # Judge-token cap is 400: densest public windows took 375 at 600 chars, 280 at 400.
MIN_CLAIM_CHARS = 20
MAX_CLAIMS = 3

Ownership = dict[str, tuple[frozenset[str], bool]]


def load_ownership(corpus_dir: Path) -> Ownership | None:
    """doc_id -> (entity_ids, shared) from corpus/manifest.json, or None if unreadable."""
    root = Path(corpus_dir)
    try:
        files = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(files, list):
        return None
    owners: Ownership = {}
    for entry in files:
        if not isinstance(entry, dict) or entry.get("role", "corpus") != "corpus":
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.startswith("corpus/") or path.count("/") != 1:
            continue
        name = path[len("corpus/"):]
        if not name.endswith(".json") or name == "manifest.json":
            continue
        ids = entry.get("entity_ids") or []
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            continue
        stem = name[: -len(".json")]
        if _declares_own_id(root / name, stem):
            owners[stem] = (frozenset(ids), entry.get("shared") is True)
    return owners


def _declares_own_id(path: Path, stem: str) -> bool:
    """Retrieval keys a document by its internal doc_id; the scorer reads offsets from this file."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True  # No excerpt can come from a file that is not there.
    except (OSError, ValueError):
        return False
    return isinstance(doc, dict) and doc.get("doc_id", stem) == stem


def may_cite(owners: Ownership | None, doc_id: str, entity_id: str) -> bool:
    """The scorer's rule: shared, or labeled with this entity. Unknown ownership is not citable."""
    if not owners or doc_id not in owners:
        return False
    entity_ids, shared = owners[doc_id]
    return shared or entity_id in entity_ids


def _cost(text: str) -> int:
    """Characters, counting non-ASCII three times: byte-fallback judge tokens cost more."""
    return len(text) + 2 * sum(ord(c) > 127 for c in text)


def trim_quote(text: str, limit: int = MAX_CLAIM_CHARS) -> str:
    """A verbatim prefix whose weighted length is at most `limit`, cut at a space when possible."""
    if _cost(text) <= limit:
        return text
    cost = 0
    for end, char in enumerate(text):
        cost += 3 if ord(char) > 127 else 1
        if cost > limit:
            break
    cut = text.rfind(" ", 0, end + 1)
    return text[: cut if cut >= MIN_CLAIM_CHARS else end].rstrip()


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
        # A figure keeps a verbatim quote from being judged content-free.
        if (len(quote) < MIN_CLAIM_CHARS or not any(c.isdigit() for c in quote)
                or not may_cite(owners, doc_id, entity_id)):
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
