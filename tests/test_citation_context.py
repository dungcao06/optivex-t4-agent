"""Citation integrity with synthetic replies, never forecast-quality evidence."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze  # Establish the pinned upstream import path.
from baselines.strong_rag_baseline.indexer import Chunk
from retrieval import _compact_tables, build_index
from runtime import normalize_prediction


TASK = {"target": {"type": "regression"}, "interval_level": 0.9}
ENTITY = {"entity_id": "one"}
TABLE = "Units: percent\nTenor | Yield\n--- | ---\n2-Year | 4.25\n10-Year | 3.75\n"
SOURCE = "Préface 😀\n" + TABLE + "After.\n"
UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))


def reply(*quotes: str, doc_id: str = "doc") -> dict:
    return {"point_forecast": 1.25, "interval": {"lo": -2, "hi": 3},
            "evidence": [{"doc_id": doc_id, "quote": quote, "claim": f"Context {i}"}
                         for i, quote in enumerate(quotes)]}


def chunk(text: str, *, start: int = 0, doc_id: str = "doc") -> Chunk:
    return Chunk(doc_id, "2024-01-01", start, start + len(text), text)


def spans(result: dict) -> list[tuple[str, int, int]]:
    return [(claim["doc_id"], claim["span_start"], claim["span_end"])
            for claim in result["claims"]]


def test_duplicate_span_with_different_prose_frees_slots_for_distinct_sources():
    """Catches deduplication by claim prose dropping later source evidence."""
    raw = reply("Alpha", "Alpha", "Alpha", "Alpha", "Beta", "Gamma", "Delta")
    result = normalize_prediction(raw, TASK, ENTITY, [chunk("Alpha Beta Gamma Delta")])
    assert spans(result) == [("doc", 0, 5), ("doc", 6, 10), ("doc", 11, 16), ("doc", 17, 22)]
    assert result["claims"][0]["claim"] == "Context 0"


def test_deduplication_does_not_scan_past_the_eighth_raw_entry():
    raw = reply(*(["Alpha"] * 8 + ["Beta"]))
    result = normalize_prediction(raw, TASK, ENTITY, [chunk("Alpha Beta")])
    assert spans(result) == [("doc", 0, 5)]


def test_same_offsets_in_different_documents_are_distinct_sources():
    raw = reply("Alpha")
    raw["evidence"].append({"doc_id": "other", "quote": "Alpha", "claim": "Other context"})
    result = normalize_prediction(raw, TASK, ENTITY,
                                  [chunk("Alpha"), chunk("Alpha", doc_id="other")])
    assert spans(result) == [("doc", 0, 5), ("other", 0, 5)]


def test_partial_row_quote_adds_exact_caption_through_row_end_with_unicode_offsets():
    """Catches byte offsets, replacement of originals, and stopping at the quote."""
    raw = reply("4.2")
    before = copy.deepcopy(raw)
    result = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)],
                                  table_spans={"doc": [(10, 78)]})
    assert spans(result) == [("doc", 58, 61), ("doc", 10, 63)]
    assert SOURCE[10:63] == "Units: percent\nTenor | Yield\n--- | ---\n2-Year | 4.25\n"
    assert result["claims"][1]["claim"] == result["claims"][0]["claim"] == "Context 0"
    assert result["point_forecast"] == 1.25
    assert result["interval"] == {"level": 0.9, "lo": -2.0, "hi": 3.0}
    assert raw == before


def test_context_with_nonzero_chunk_offset_preserves_crlf_characters():
    table = TABLE.replace("\n", "\r\n")
    result = normalize_prediction(reply("4.2"), TASK, ENTITY, [chunk(table, start=20)],
                                  table_spans={"doc": [(20, 93)]})
    assert spans(result) == [("doc", 71, 74), ("doc", 20, 77)]


def test_all_originals_precede_at_most_one_context_addition():
    result = normalize_prediction(reply("4.25", "3.75"), TASK, ENTITY,
                                  [chunk(SOURCE)], table_spans={"doc": [(10, 78)]})
    assert spans(result) == [("doc", 58, 62), ("doc", 73, 77), ("doc", 10, 63)]


def test_four_unique_originals_leave_no_room_for_context():
    raw = reply("4.25", "3.75", "After.", "Préface", "Yield")
    original = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)])
    enriched = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)],
                                    table_spans={"doc": [(10, 78)]})
    assert spans(enriched) == [("doc", 58, 62), ("doc", 73, 77), ("doc", 78, 84), ("doc", 0, 7)]
    assert enriched == original


def test_duplicate_originals_free_one_slot_without_displacing_later_originals():
    raw = reply("4.25", "4.25", "4.25", "4.25", "3.75", "After.")
    result = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)],
                                  table_spans={"doc": [(10, 78)]})
    assert spans(result) == [("doc", 58, 62), ("doc", 73, 77), ("doc", 78, 84), ("doc", 10, 63)]


@pytest.mark.parametrize("metadata", [
    None, {}, [], {"other": [(10, 78)]}, {"doc": None}, {"doc": "bad"},
    {"doc": [None, 1, (), (10,), (10, 78, 90), ("10", 78), (True, 78),
             (-1, 78), (78, 10), (10, 9000), (10, float("inf"))]},
])
def test_missing_or_malformed_optional_metadata_keeps_valid_original(metadata):
    result = normalize_prediction(reply("4.25"), TASK, ENTITY, [chunk(SOURCE)],
                                  table_spans=metadata)
    assert spans(result) == [("doc", 58, 62)]


@pytest.mark.parametrize("start,end", [(10, 63), (39, 78)])
def test_full_original_table_must_fit_one_visible_chunk(start, end):
    result = normalize_prediction(reply("4.25"), TASK, ENTITY,
                                  [chunk(SOURCE[start:end], start=start)],
                                  table_spans={"doc": [(10, 78)]})
    assert spans(result) == [("doc", 58, 62)]


def test_unrecognized_table_does_not_gain_context():
    text = "Tenor | Yield\n2-Year | 4.25 | uncertain\n"
    result = normalize_prediction(reply("4.25"), TASK, ENTITY, [chunk(text)],
                                  table_spans={"doc": _compact_tables(text)})
    assert len(result["claims"]) == 1


def test_oversized_original_table_cannot_expand_a_recognizable_clipped_excerpt():
    text = TABLE + "20-Year | 4.10\n" * 200
    excerpt = chunk(TABLE)
    assert _compact_tables(excerpt.text)  # The dangerous excerpt-only interpretation.
    assert _compact_tables(text) == []
    result = normalize_prediction(reply("4.25"), TASK, ENTITY, [excerpt],
                                  table_spans={"doc": _compact_tables(text)})
    assert spans(result) == [("doc", 48, 52)]


def test_unavailable_document_quote_cannot_create_a_new_source():
    raw = reply("4.25")
    raw["evidence"].append({"doc_id": "hidden", "quote": "3.75", "claim": "Hidden context"})
    result = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)],
                                  table_spans={"doc": [(10, 78)], "hidden": [(10, 78)]})
    assert spans(result) == [("doc", 58, 62), ("doc", 10, 63)]


def test_context_equal_to_an_existing_original_is_not_duplicated():
    text = SOURCE[:62]
    raw = reply("4.25", text[10:])
    result = normalize_prediction(raw, TASK, ENTITY, [chunk(text)],
                                  table_spans={"doc": [(10, 62)]})
    assert spans(result) == [("doc", 58, 62), ("doc", 10, 62)]


@pytest.mark.parametrize("quote", ["Units: percent", "Tenor", "---"])
def test_caption_header_or_separator_alone_does_not_gain_a_data_row(quote):
    result = normalize_prediction(reply(quote), TASK, ENTITY, [chunk(SOURCE)],
                                  table_spans={"doc": [(10, 78)]})
    assert len(result["claims"]) == 1


def test_even_invalid_metadata_cannot_expand_over_the_passage_limit():
    text = TABLE + "20-Year | 4.10\n" * 200
    result = normalize_prediction(reply("4.25"), TASK, ENTITY, [chunk(text)],
                                  table_spans={"doc": [(0, len(text))]})
    assert spans(result) == [("doc", 48, 52)]


def test_synthetic_zero_support_context_keeps_the_original_support_input():
    """Integrity only: this dictionary is not a production entailment model."""
    raw = reply("4.25")
    original = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)])
    enriched = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)],
                                    table_spans={"doc": [(10, 78)]})
    support = {("doc", 58, 62): 0.8, ("doc", 10, 63): 0.0}
    assert len(enriched["claims"]) == 2
    assert max(support[span] for span in spans(original)) == 0.8
    assert max(support[span] for span in spans(enriched)) == 0.8


def test_published_verifier_can_fail_on_added_context_after_original_support():
    """Adding a valid span is not operationally monotonic in the real primitive."""
    from qfbench2_common.scoring.faithfulness import citation_faithfulness

    class FailingContextJudge:
        def entail(self, premise, hypothesis):
            if premise == "4.25":
                return 0.8
            raise TimeoutError("synthetic context evaluation failure")

    raw = reply("4.25")
    original = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)])
    enriched = normalize_prediction(raw, TASK, ENTITY, [chunk(SOURCE)],
                                    table_spans={"doc": [(10, 78)]})
    original_claim = {"text": "Synthetic prediction hypothesis", "citations": original["claims"]}
    enriched_claim = dict(original_claim, citations=enriched["claims"])
    lookup = {"doc": SOURCE}.__getitem__
    judge = FailingContextJudge()
    assert citation_faithfulness([original_claim], lookup, judge) == 1.0
    with pytest.raises(TimeoutError, match="synthetic context evaluation failure"):
        citation_faithfulness([enriched_claim], lookup, judge)


def test_public_rates_last_row_preserves_original_and_complete_table():
    unit = UPSTREAM / "units" / "t4-fomc-curve-20240918"
    corpus = build_index(unit / "corpus")
    doc_id = "RATES_SNAPSHOT_20240919"
    text = corpus.doc_texts[doc_id]
    raw = reply(text[2381:2446], doc_id=doc_id)
    result = normalize_prediction(raw, TASK, ENTITY,
                                  [part for part in corpus.chunks if part.doc_id == doc_id],
                                  table_spans={doc_id: _compact_tables(text)})
    assert spans(result) == [(doc_id, 2381, 2446), (doc_id, 1592, 2447)]


def test_cli_wires_original_table_metadata_without_additional_model_requests(tmp_path, monkeypatch):
    """A real CLI subprocess and local HTTP catch a missing production callsite."""
    import http_fixture
    from test_http_runtime import run_agent

    unit = UPSTREAM / "units" / "t4-fomc-curve-20240918"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    task["entities"] = task["entities"][:1]
    raw = reply("2024-09-19 | 3.93 | 3.59 | 3.47 | 3.49 | 3.6 | 3.73 | 4.11 | 4.06", doc_id="RATES_SNAPSHOT_20240919")
    monkeypatch.setattr(http_fixture, "prompt_reply", lambda payload, mode: json.dumps(raw))
    answer, requests = run_agent(tmp_path, unit, task=task)
    assert answer["notes"]["degraded_entities"] == 0
    assert answer["notes"]["model_requests"] == len(requests) == 1
    assert spans(answer["entity_predictions"][0]) == [
        ("RATES_SNAPSHOT_20240919", 2381, 2446), ("RATES_SNAPSHOT_20240919", 1592, 2447)]


def test_run_recognizes_each_retrieved_original_document_once(tmp_path, monkeypatch):
    """Catches repeated document parsing across entities or quotation retries."""
    import http_fixture
    import runtime

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "source.json").write_text(json.dumps(
        {"doc_id": "doc", "doc_date": "2024-01-01", "text": SOURCE}), encoding="utf-8")
    task = dict(TASK, task_id="synthetic", cutoff_date="2024-01-31",
                entities=[ENTITY, {"entity_id": "two"}])
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    calls = []

    def recognize(text):
        calls.append(text)
        return _compact_tables(text)

    monkeypatch.setattr(runtime, "_compact_tables", recognize, raising=False)
    monkeypatch.setattr(http_fixture, "prompt_reply", lambda payload, mode: json.dumps(reply("4.25")))
    with http_fixture.model_server() as (endpoint, requests):
        monkeypatch.setenv("MODEL_ENDPOINT", endpoint)
        answer = runtime.run(task_path, corpus_dir, tmp_path / "answer.json",
                             analyze.SYSTEM_PROMPT, analyze.optivex_prompt)
    assert answer["notes"]["degraded_entities"] == 0
    assert len(requests) == 2
    assert [spans(row) for row in answer["entity_predictions"]] == [
        [("doc", 58, 62), ("doc", 10, 63)], [("doc", 58, 62), ("doc", 10, 63)]]
    assert calls == [SOURCE]
