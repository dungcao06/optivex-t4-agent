# Increment A (Claude part): citable claims and evidence-based fallback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every claim the agent writes one that Track 4 scorer 5.2.2 cannot mark false, and replace the unconditional 0 / [-1, 1] fallback row with one derived from the unit's successful rows.

**Architecture:** A new stdlib-only `claims.py` reads document ownership from the read-only `corpus/manifest.json`, keeps only short verbatim model quotes from documents the entity may cite, and falls back to a quote of the entity's own task-table row (`doc_id: "task"`). A new `fallback.py` builds failed rows from the median of successful rows. `runtime.py` drops the scorer-3.1.0 table-context enrichment and wires both modules in. Upstream pins move to public main ede7381 (scorer 5.2.2) and toolkit v2.5.1.

**Tech Stack:** Python 3.13 stdlib only at runtime; pytest 9.1.1, jsonschema 4.26.0 and qfbench2-common 2.5.1 for tests; the public repo at ede7381 provides scorer parity checks (`qfbench2_track_analysis`, `baselines/guardrails_example/citation_rail.py`).

## Global Constraints

- Source of design: `agenthon-2026/top1-pathway.md` revision 2 (Codex sign-off 13:20, user approval 13:16, 1 Oct 2026).
- Scorer 5.2.2 claim rules: false if the cited document is not labeled for the entity nor shared (task spans must lie inside the entity's own row), offsets out of range, claim text empty or over 4,000 characters or 400 judge tokens, a citation over 8,000 characters, a figure no cited span carries, content-free, or contradiction probability above 0.9. Unlabelled documents are citable by no one (`corpus.py:136-143`).
- Task table text (`qfbench2_track_analysis/corpus.py:68-77`): one `json.dumps(row, ensure_ascii=False, separators=(", ", ": "))` line per `entities` row, joined by `"\n"`, no trailing newline.
- `claims` must have at least 1 item per entity (`analysis.schema.json`).
- Runtime code imports nothing from the scorer; the image contains only `baselines/strong_rag_baseline` from upstream.
- Pins: upstream `ede7381d8c1ba9d8c84068f9d142f5e093a33892`; qfbench2-common v2.5.1 = commit `50fb2dc2b39c70f4cf81fcd269943782eddfaed0`.
- Do not edit `retrieval.py`, `analyze.py` or any target-record module (Codex's part of Increment A).
- Local test command (old checkout stays untouched):
  `T4_UPSTREAM_PATH=../track4-analysis-public-ede7381 <venv>/bin/python -m pytest tests -q`, run from `optivex-t4-agent/`, with `<venv>` = a Python 3.13 environment holding pytest 9.1.1, jsonschema 4.26.0 and qfbench2-common 2.5.1.

---

### Task 1: Move upstream pins to scorer 5.2.2 and confirm the existing suite

**Files:**
- Modify: `.github/workflows/publish-image.yml` (upstream `ref`, qfbench2-common pin)
- Modify: `Dockerfile:3` (`TRACK4_COMMIT`)

**Interfaces:**
- Consumes: nothing.
- Produces: all later tasks run tests against ede7381 units.

- [ ] **Step 1: Update pins**

In `.github/workflows/publish-image.yml`, every `ref: 2b307560c8183905a030dcb4cd26ce857a039cfd` becomes `ref: ede7381d8c1ba9d8c84068f9d142f5e093a33892`, and
`git+https://github.com/Agenthon-2026/Agenthon2026-public.git@68bc878a740eb0040e70ddf0feab98405dca1d6d#subdirectory=common` becomes
`git+https://github.com/Agenthon-2026/Agenthon2026-public.git@50fb2dc2b39c70f4cf81fcd269943782eddfaed0#subdirectory=common`.
In `Dockerfile`, `ARG TRACK4_COMMIT=2b307560c8183905a030dcb4cd26ce857a039cfd` becomes `ARG TRACK4_COMMIT=ede7381d8c1ba9d8c84068f9d142f5e093a33892`.

- [ ] **Step 2: Install the test environment and run the unchanged suite against ede7381**

Run: `uv pip install --python <venv>/bin/python "pytest==9.1.1" "jsonschema==4.26.0"` then the local test command.
Expected: the suite passes, or any failures are caused only by the new units' metadata (record them; they are fixed in Task 5). The scaffold prompt change at ede7381 only rewords the ranking instruction and drops `rank`, so `analyze.py`'s `.replace` calls become harmless no-ops.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish-image.yml Dockerfile
git commit -m "build: pin Track 4 upstream to scorer 5.2.2 and toolkit 2.5.1"
```

---

### Task 2: Ownership and task-row claims (`claims.py`, part 1)

**Files:**
- Create: `claims.py`
- Test: `tests/test_claims.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Ownership = dict[str, tuple[frozenset[str], bool]]`; `load_ownership(corpus_dir: Path) -> Ownership | None`; `may_cite(owners: Ownership | None, doc_id: str, entity_id: str) -> bool`; `trim_quote(text: str, limit: int = MAX_CLAIM_CHARS) -> str`; `task_row(task: dict, entity_id: str) -> tuple[str, int] | None`; `task_row_claim(task: dict, entity_id: str) -> dict | None`; constants `TASK_DOC_ID = "task"`, `MAX_CLAIM_CHARS = 600`, `MIN_CLAIM_CHARS = 20`, `MAX_CLAIMS = 3`.

- [ ] **Step 1: Write the failing tests**

```python
"""Claims scorer 5.2.2 cannot mark false: ownership, task rows, verbatim quotes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze  # noqa: F401  Establish the pinned upstream import path.
from claims import (MAX_CLAIM_CHARS, TASK_DOC_ID, load_ownership, may_cite, task_row,
                    task_row_claim, trim_quote)

UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))
UNITS = sorted(p.parent for p in (UPSTREAM / "units").glob("*/task.json")
               if (p.parent / "corpus" / "manifest.json").is_file())


def write_manifest(tmp_path: Path, files: list[dict]) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps({"files": files}))
    return tmp_path


def test_ownership_reads_labels_shared_and_unlabelled(tmp_path):
    corpus = write_manifest(tmp_path, [
        {"path": "corpus/A.json", "entity_ids": ["x"]},
        {"path": "corpus/S.json", "shared": True},
        {"path": "corpus/U.json"},
        {"path": "corpus/manifest.json"},
    ])
    owners = load_ownership(corpus)
    assert may_cite(owners, "A", "x") and not may_cite(owners, "A", "y")
    assert may_cite(owners, "S", "y")
    assert not may_cite(owners, "U", "x")  # unlabelled: citable by no one, as in the scorer
    assert "manifest" not in owners


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"files": "x"})])
def test_missing_or_broken_manifest_means_nothing_is_citable(tmp_path, content):
    if content is not None:
        (tmp_path / "manifest.json").write_text(content)
    owners = load_ownership(tmp_path)
    assert owners is None and not may_cite(owners, "A", "x")


def test_trim_quote_keeps_a_verbatim_prefix():
    text = "word " * 300
    trimmed = trim_quote(text)
    assert len(trimmed) <= MAX_CLAIM_CHARS and text.startswith(trimmed)
    assert trim_quote("short quote stays whole") == "short quote stays whole"


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_task_rows_match_the_scorer_task_table(unit):
    from qfbench2_track_analysis.corpus import task_table_text
    task = json.loads((unit / "task.json").read_text())
    text, ranges = task_table_text(task)
    for entity in task["entities"]:
        line, start = task_row(task, entity["entity_id"])
        assert (start, start + len(line)) == ranges[entity["entity_id"]]
        assert text[start:start + len(line)] == line
        claim = task_row_claim(task, entity["entity_id"])
        assert claim["doc_id"] == TASK_DOC_ID
        assert text[claim["span_start"]:claim["span_end"]] == claim["claim"]


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_ownership_matches_the_scorer_on_every_public_document(unit):
    from qfbench2_track_analysis.corpus import CorpusIndex
    owners = load_ownership(unit / "corpus")
    index = CorpusIndex.from_unit(unit)
    task = json.loads((unit / "task.json").read_text())
    for doc_id, doc in index._docs.items():
        for entity in task["entities"]:
            assert may_cite(owners, doc_id, entity["entity_id"]) == doc.admits(entity["entity_id"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: local test command with `tests/test_claims.py`.
Expected: FAIL with `ModuleNotFoundError: No module named 'claims'`.

- [ ] **Step 3: Write the implementation**

```python
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
MAX_CLAIM_CHARS = 600   # Far below the 4,000-character and 400-judge-token claim limits.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: local test command with `tests/test_claims.py`.
Expected: all pass. If `test_ownership_matches_the_scorer_on_every_public_document` fails, the doc_id derivation differs from the scorer's `_doc_id_from_path`; fix `load_ownership`, not the test.

- [ ] **Step 5: Commit**

```bash
git add claims.py tests/test_claims.py
git commit -m "feat(claims): read citation ownership and task-row quotes"
```

---

### Task 3: Verbatim citable claims and prompt citation rule (`claims.py`, part 2)

**Files:**
- Modify: `claims.py` (append `build_claims`, `citable_note`)
- Test: `tests/test_claims.py` (append)

**Interfaces:**
- Consumes: Task 2 functions and constants.
- Produces: `build_claims(items: object, chunks: list, task: dict, entity_id: str, owners: Ownership | None) -> list[dict]`; `citable_note(owners: Ownership | None, entity_id: str, chunks: list) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
from baselines.strong_rag_baseline.indexer import Chunk
from claims import MAX_CLAIMS, build_claims, citable_note

TASK = {"target": {"type": "regression"}, "interval_level": 0.9,
        "entities": [{"entity_id": "one", "start_yield_pct": 3.73}, {"entity_id": "two"}]}
OWNERS = {"doc": (frozenset({"one"}), False), "peer": (frozenset({"two"}), False),
          "shared": (frozenset(), True)}
S1, S2, S3, S4 = ("Alpha yield rose to 4.25 percent.", "Beta spread was 12 basis points.",
                  "Gamma auction drew 2.61 times.", "Delta supply grew 7 percent.")
TEXT = " ".join([S1, S2, S3, S4])


def chunk(text: str, *, start: int = 0, doc_id: str = "doc") -> Chunk:
    return Chunk(doc_id, "2024-01-01", start, start + len(text), text)


def item(quote: str, doc_id: str = "doc") -> dict:
    return {"doc_id": doc_id, "quote": quote, "claim": "model prose with 999 percent"}


def test_claim_text_is_the_exact_quote_not_model_prose():
    claims = build_claims([item(S1)], [chunk(TEXT, start=100)], TASK, "one", OWNERS)
    assert claims == [{"doc_id": "doc", "span_start": 100, "span_end": 100 + len(S1), "claim": S1}]


def test_peer_documents_are_never_cited_and_shared_ones_are():
    chunks = [chunk(TEXT), chunk(TEXT, doc_id="peer"), chunk(TEXT, doc_id="shared")]
    claims = build_claims([item(S1, "peer"), item(S2, "shared")], chunks, TASK, "one", OWNERS)
    assert [c["doc_id"] for c in claims] == ["shared"]


def test_duplicates_collapse_and_at_most_three_claims_survive():
    raw = [item(S1), item(S1), item(S2), item(S3), item(S4)]
    claims = build_claims(raw, [chunk(TEXT)], TASK, "one", OWNERS)
    assert [c["claim"] for c in claims] == [S1, S2, S3] and len(claims) == MAX_CLAIMS


def test_only_the_first_eight_raw_items_are_read():
    claims = build_claims([item(S1)] * 8 + [item(S2)], [chunk(TEXT)], TASK, "one", OWNERS)
    assert [c["claim"] for c in claims] == [S1]


@pytest.mark.parametrize("raw", [None, "text", [], [item("not in the excerpt at all, 5 percent")],
                                 [item("tiny")], [{"doc_id": ["doc"], "quote": S1}]])
def test_unusable_evidence_falls_back_to_the_entity_task_row(raw):
    claims = build_claims(raw, [chunk(TEXT)], TASK, "one", OWNERS)
    assert claims == [task_row_claim(TASK, "one")]


def test_unknown_ownership_cites_only_the_task_row():
    claims = build_claims([item(S1)], [chunk(TEXT)], TASK, "one", None)
    assert claims == [task_row_claim(TASK, "one")]


def test_long_quotes_are_trimmed_to_a_verbatim_prefix():
    long = "Alpha yield rose " + "and kept rising " * 60 + "to 4.25 percent."
    claims = build_claims([item(long)], [chunk(long)], TASK, "one", OWNERS)
    assert len(claims[0]["claim"]) <= 600 and long.startswith(claims[0]["claim"])
    assert claims[0]["span_end"] - claims[0]["span_start"] == len(claims[0]["claim"])


def test_citable_note_names_only_citable_documents():
    chunks = [chunk(TEXT), chunk(TEXT, doc_id="peer"), chunk(TEXT, doc_id="shared")]
    note = citable_note(OWNERS, "one", chunks)
    assert "doc, shared" in note and "peer" not in note
    assert '"evidence": []' in citable_note(OWNERS, "one", [chunk(TEXT, doc_id="peer")])


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_scorer_finds_no_false_claim_when_the_model_quotes_every_excerpt(unit):
    """The model may quote any excerpt; build_claims must leave only claims the scorer accepts."""
    from baselines.guardrails_example.citation_rail import check_claim_rules
    from retrieval import EvidenceIndex, build_index
    task = json.loads((unit / "task.json").read_text())
    owners = load_ownership(unit / "corpus")
    index = EvidenceIndex(build_index(unit / "corpus"), task["cutoff_date"])
    rows = []
    for entity in task["entities"]:
        chunks = index.retrieve(task, entity, top_k=8)
        raw = [item(c.text[:240].strip(), c.doc_id) for c in chunks]
        row = {"entity_id": entity["entity_id"], "point_forecast": 0.0,
               "interval": {"level": task["interval_level"], "lo": -1.0, "hi": 1.0},
               "claims": build_claims(raw, chunks, task, entity["entity_id"], owners)}
        if task["target"]["type"] == "classification":
            row["label"] = task["target"]["labels"][0]
        rows.append(row)
    answer = {"task_id": task["task_id"], "schema_version": "3", "entity_predictions": rows}
    findings = [f for f in check_claim_rules(answer, unit, token_counter=None)
                if f.code != "claim_tokens_unchecked"]
    assert findings == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: local test command with `tests/test_claims.py`.
Expected: FAIL with `ImportError: cannot import name 'build_claims'`.

- [ ] **Step 3: Write the implementation (append to `claims.py`)**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: local test command with `tests/test_claims.py`.
Expected: all pass, including the scorer parity test on all 10 manifest-bearing units.

- [ ] **Step 5: Commit**

```bash
git add claims.py tests/test_claims.py
git commit -m "feat(claims): keep only verbatim quotes the entity may cite"
```

---

### Task 4: Evidence-based fallback rows (`fallback.py`)

**Files:**
- Create: `fallback.py`
- Test: `tests/test_fallback.py`

**Interfaces:**
- Consumes: nothing (claims are passed in).
- Produces: `fallback_prediction(task: dict, entity: dict, peers: list[dict], claims: list[dict]) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
"""Fallback rows come from the unit's successful rows, never an unconditional placeholder."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fallback import fallback_prediction

CLAIM = [{"doc_id": "task", "span_start": 0, "span_end": 25, "claim": '{"entity_id": "c", "x": 1}'}]
REG = {"target": {"type": "regression"}, "interval_level": 0.9}
CLS = {"target": {"type": "classification", "labels": ["up", "down"]}, "interval_level": 0.9}


def peer(point, lo, hi, label=None):
    row = {"point_forecast": point, "interval": {"level": 0.9, "lo": lo, "hi": hi}}
    if label:
        row["label"] = label
    return row


def test_regression_uses_peer_medians_and_contains_the_point():
    row = fallback_prediction(REG, {"entity_id": "c"},
                              [peer(2.4, 2.1, 2.7), peer(2.6, 2.3, 2.9), peer(9.0, 8.0, 10.0)], CLAIM)
    assert row["point_forecast"] == 2.6
    assert row["interval"] == {"level": 0.9, "lo": 2.3, "hi": 2.9}
    assert row["claims"] == CLAIM and row["entity_id"] == "c"


def test_classification_uses_the_peer_majority_label():
    peers = [peer(0.7, 0.5, 0.9, "up"), peer(0.6, 0.4, 0.8, "up"), peer(0.2, 0.1, 0.4, "down")]
    row = fallback_prediction(CLS, {"entity_id": "c"}, peers, CLAIM)
    assert row["label"] == "up" and row["point_forecast"] == 0.6


def test_without_peers_the_legacy_values_remain():
    row = fallback_prediction(CLS, {"entity_id": "c"}, [], CLAIM)
    assert row["label"] == "up" and row["point_forecast"] == 0.0
    assert row["interval"] == {"level": 0.9, "lo": -1.0, "hi": 1.0}


def test_interval_always_contains_the_point():
    row = fallback_prediction(REG, {"entity_id": "c"}, [peer(5.0, 0.0, 1.0), peer(6.0, 0.0, 1.0)], CLAIM)
    assert row["interval"]["lo"] <= row["point_forecast"] <= row["interval"]["hi"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: local test command with `tests/test_fallback.py`.
Expected: FAIL with `ModuleNotFoundError: No module named 'fallback'`.

- [ ] **Step 3: Write the implementation**

```python
"""Fallback rows when inference fails, derived from the unit's successful rows.

Replaces the unconditional point 0 / interval [-1, 1] placeholder. The peer median is a
cross-entity estimate, not evidence about this entity; run notes still mark the row unverified.
"""
from __future__ import annotations

import statistics
from collections import Counter

LEGACY_LABELS = ("inline", "neutral", "unchanged")


def fallback_prediction(task: dict, entity: dict, peers: list[dict], claims: list[dict]) -> dict:
    values = [p["point_forecast"] for p in peers if "point_forecast" in p]
    if values:
        point = float(statistics.median(values))
        lo = min(float(statistics.median([p["interval"]["lo"] for p in peers])), point)
        hi = max(float(statistics.median([p["interval"]["hi"] for p in peers])), point)
    else:
        point, lo, hi = 0.0, -1.0, 1.0
    row = {"entity_id": entity["entity_id"], "point_forecast": point,
           "interval": {"level": task.get("interval_level", 0.9), "lo": lo, "hi": hi},
           "claims": claims}
    target = task["target"]
    if target["type"] == "classification":
        labels = [p["label"] for p in peers if p.get("label") in target["labels"]]
        row["label"] = (Counter(labels).most_common(1)[0][0] if labels else
                        next((x for x in LEGACY_LABELS if x in target["labels"]), target["labels"][0]))
    return row
```

- [ ] **Step 4: Run tests to verify they pass**

Run: local test command with `tests/test_fallback.py`.
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add fallback.py tests/test_fallback.py
git commit -m "feat(fallback): derive failed rows from the unit's successful rows"
```

---

### Task 5: Integrate into the runtime and retire table-context enrichment

**Files:**
- Modify: `runtime.py` (imports; delete `_table_context` and `degraded_prediction`; `normalize_prediction`; `run`)
- Delete: `tests/test_citation_context.py` (scorer-3.1.0 enrichment; dedup/offset cases now live in `tests/test_claims.py`)
- Modify: `tests/test_http_runtime.py` (`assert_answer`, `test_ungrounded_quote_cannot_keep_its_claim`)
- Modify: `.github/workflows/publish-image.yml` (container test command), `Dockerfile` (COPY new modules)

**Interfaces:**
- Consumes: `build_claims`, `citable_note`, `load_ownership`, `task_row_claim` (Tasks 2-3); `fallback_prediction` (Task 4).
- Produces: `normalize_prediction(raw: dict, task: dict, entity: dict, chunks: list, *, owners: Ownership | None = None) -> dict` (the `table_spans` keyword is removed).

- [ ] **Step 1: Write the failing tests (edit `tests/test_http_runtime.py`)**

Replace the claim loop at the end of `assert_answer` (currently lines 88-91) with a scorer check, and add `unit` claim rules:

```python
        for claim in row["claims"]:
            if claim["doc_id"] == "task":
                continue  # Checked against the entity's own row by the scorer rules below.
            doc = corpus.doc_texts[claim["doc_id"]]
            assert 0 <= claim["span_start"] < claim["span_end"] <= len(doc)
            assert corpus.doc_dates[claim["doc_id"]] <= task["cutoff_date"]
    if (unit / "corpus" / "manifest.json").is_file() or (unit / "manifest.json").is_file():
        from baselines.guardrails_example.citation_rail import check_claim_rules
        findings = [f for f in check_claim_rules(answer, unit, token_counter=None)
                    if f.code != "claim_tokens_unchecked"]
        assert findings == [], findings
```

Replace `test_ungrounded_quote_cannot_keep_its_claim` with:

```python
def test_ungrounded_quote_cannot_keep_its_claim(tmp_path):
    answer, requests = run_agent(tmp_path, EXAMPLE, "ungrounded")
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert UNSUPPORTED_CLAIM not in json.dumps(answer)
    assert answer.get("notes", {}).get("degraded_entities", 0) == 0
    assert [claim["doc_id"] for row in answer["entity_predictions"] for claim in row["claims"]] == ["task"]
    assert_request_contract(requests)
```

Delete `tests/test_citation_context.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: local test command with `tests/test_http_runtime.py`.
Expected: FAIL. `test_ungrounded_quote_cannot_keep_its_claim` still degrades (old runtime), and units whose mock quotes peer documents report `claim_wrong_entity` or task-free claims.

- [ ] **Step 3: Write the implementation in `runtime.py`**

Imports: replace `from retrieval import EvidenceIndex, _MAX_PASSAGE, _compact_tables, build_index` with

```python
from claims import Ownership, build_claims, citable_note, load_ownership, task_row_claim
from fallback import fallback_prediction
from retrieval import EvidenceIndex, build_index
```

Delete `_table_context` and `degraded_prediction` entirely. In `normalize_prediction`, change the signature to `def normalize_prediction(raw: dict, task: dict, entity: dict, chunks: list, *, owners: Ownership | None = None) -> dict:` and replace everything from `items = raw.get("evidence")` to the end of the function with:

```python
    claims = build_claims(raw.get("evidence"), chunks, task, entity["entity_id"], owners)
    if not claims:
        raise ValueError("No citable claim for this entity")
    prediction["claims"] = claims
    return prediction
```

In `run`, replace the lines from `retrieved_docs = ...` through `prompts = [...]` with:

```python
    owners = load_ownership(corpus_dir)
    prompts = [prompt_builder(task, entity, context) + citable_note(owners, entity["entity_id"], context)
               for entity, context in zip(entities, contexts)]
```

and the `normalize_prediction(...)` call's keyword `table_spans=table_spans` with `owners=owners`. Replace the degraded loop with:

```python
    degraded_ids = []
    peers = [prediction for prediction in predictions if prediction is not None]
    for i, prediction in enumerate(predictions):
        if prediction is None:
            degraded_ids.append(entities[i]["entity_id"])
            claim = task_row_claim(task, entities[i]["entity_id"])
            predictions[i] = fallback_prediction(task, entities[i], peers, [claim] if claim else [])
```

and the `quality_warning` text with `"Inference failed for these entities. Their forecasts are the median of this unit's successful rows and cite only the entity's task row; they are not evidence of predictive quality."`.

In `Dockerfile`, change `COPY runtime.py retrieval.py /app/` to `COPY runtime.py retrieval.py claims.py fallback.py /app/`.
In `.github/workflows/publish-image.yml`, the container test `run:` becomes `python -m pytest tests/test_http_runtime.py -q`.

- [ ] **Step 4: Run the full suite**

Run: local test command (all tests).
Expected: all pass. `grep -n "_table_context\|table_spans\|degraded_prediction" runtime.py tests/*.py` prints nothing.

- [ ] **Step 5: Commit**

```bash
git add runtime.py Dockerfile .github/workflows/publish-image.yml tests/test_http_runtime.py
git rm tests/test_citation_context.py
git commit -m "feat(runtime): write only citable claims and evidence-based fallbacks"
```

---

### Task 6: End-to-end verification, README note and review

**Files:**
- Modify: `README.md` (replace the candidate C citation-enrichment note with a 5.2.2 claims note)

- [ ] **Step 1: Mock run on all 11 units plus the scorer's claim rules**

Run candidate output for each unit with `--mock` (as in Step 0 of the pathway) into a scratch directory and apply `check_claim_rules` and `claim_penalty_preview`.
Expected: `refused=False`, `false=0`, `factor=1.0` on all 11; only `claim_tokens_unchecked` findings.

- [ ] **Step 2: Check the 400-judge-token cap with the judge tokenizers only**

Run: `uv pip install --python <venv>/bin/python transformers tokenizers sentencepiece protobuf`, then `AutoTokenizer.from_pretrained` for `cross-encoder/nli-deberta-v3-large` and `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (tokenizer files only, no weights), and rerun Step 1 with `token_counter=judge_token_counter()`.
Expected: no `claim_malformed` findings; record the maximum claim token count.

- [ ] **Step 3: Update `README.md`**

Replace the candidate C paragraph on citation enrichment with: "Scorer 5.2.2 (public main ede7381) charges false claims and pays nothing for true ones. Claims are therefore short verbatim quotes from documents the corpus manifest labels for the entity or marks shared, or the entity's own task-table row. Peer documents remain readable context. Failed rows use the median of the unit's successful rows. These checks establish claim validity, not forecast quality."

- [ ] **Step 4: Commit, then request review from Codex via COLLAB.md**

```bash
git add README.md
git commit -m "docs: describe scorer 5.2.2 claim policy"
```

Post the diff range and test counts in `COLLAB.md`; Codex reviews before any CI publish or upload. Push, PR, image publish and any Development upload each need the user's go-ahead.

---

## Execution notes (1 October 2026)

- Tasks 1-6 executed inline on branch `feat/scorer-522-claims` (from origin/main b897309).
- Deviation: `MAX_CLAIM_CHARS` lowered from 600 to 400. With the judge tokenizers, the densest 600-character public corpus window measured 375 judge tokens (cap 400); a 400-character window measures 280.
- Task 5 also adjusted `tests/test_contract.py` and `tests/test_response.py` (their minimal tasks gained an `entities` row so a task-row claim exists) and moved the `doc_id_list` case into its own test: a malformed citation now keeps the forecast.
- Results: 205 tests pass against ede7381; mock outputs on all 11 units have 0 findings under the scorer's deterministic claim rules including the token cap (factor 1.0); stress case quoting all 558 retrieved excerpts: 0 findings (was 142 `wrong_entity`).
