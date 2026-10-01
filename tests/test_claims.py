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


from baselines.strong_rag_baseline.indexer import Chunk  # noqa: E402
from claims import MAX_CLAIMS, build_claims, citable_note  # noqa: E402

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
    long = "Alpha yield rose 4.25 percent " + "and kept rising " * 60 + "to the end."
    claims = build_claims([item(long)], [chunk(long)], TASK, "one", OWNERS)
    assert len(claims[0]["claim"]) <= MAX_CLAIM_CHARS and long.startswith(claims[0]["claim"])
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


def test_quote_without_a_figure_is_not_cited():
    """A figure-free verbatim quote can still be judged content-free by the scorer."""
    words = "The passage discusses the evidence in general terms only."
    claims = build_claims([item(words)], [chunk(words + " " + TEXT)], TASK, "one", OWNERS)
    assert claims == [task_row_claim(TASK, "one")]


def test_non_ascii_text_is_trimmed_by_weighted_length():
    text = "Yield 4.25 percent " + "\u2014\u00e9 " * 300
    trimmed = trim_quote(text)
    assert text.startswith(trimmed)
    assert len(trimmed) + 2 * sum(ord(c) > 127 for c in trimmed) <= MAX_CLAIM_CHARS


def test_shared_document_with_null_entity_ids_is_citable(tmp_path):
    owners = load_ownership(write_manifest(tmp_path, [
        {"path": "corpus/S.json", "shared": True, "entity_ids": None}]))
    assert may_cite(owners, "S", "anyone")


def test_non_corpus_roles_and_paths_are_ignored(tmp_path):
    owners = load_ownership(write_manifest(tmp_path, [
        {"path": "corpus/R.json", "role": "reference", "entity_ids": ["x"]},
        {"path": "other/O.json", "entity_ids": ["x"]}]))
    assert not may_cite(owners, "R", "x") and not may_cite(owners, "O", "x")


def test_document_whose_internal_id_differs_from_its_file_name_is_not_citable(tmp_path):
    """Its offsets would be read against a different file by the scorer."""
    write_manifest(tmp_path, [{"path": "corpus/A.json", "entity_ids": ["x"]},
                              {"path": "corpus/B.json", "entity_ids": ["x"]}])
    (tmp_path / "A.json").write_text(json.dumps({"doc_id": "B", "text": "alpha"}))
    (tmp_path / "B.json").write_text(json.dumps({"doc_id": "B", "text": "beta"}))
    owners = load_ownership(tmp_path)
    assert not may_cite(owners, "A", "x") and may_cite(owners, "B", "x")


def test_entity_identifier_digits_are_not_a_content_figure():
    text = 'Available evidence for UST10Y.'
    task = {'entities': [{'entity_id': 'UST10Y'}]}
    owners = {'doc': (frozenset({'UST10Y'}), False)}
    claims = build_claims([item(text)], [chunk(text)], task, 'UST10Y', owners)
    assert claims == [task_row_claim(task, 'UST10Y')]
