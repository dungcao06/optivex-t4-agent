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
