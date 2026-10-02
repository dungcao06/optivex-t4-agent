"""Deterministic history records from frozen tables: exact provenance, no forecasting."""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze  # noqa: F401  Establish the pinned upstream import path.
from baselines.strong_rag_baseline.indexer import Chunk
from claims import load_ownership, may_cite
from history import build_history, format_history

UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))
UNITS = sorted(p.parent for p in (UPSTREAM / "units").glob("t4-[a-z]*/task.json")
               if (p.parent / "corpus" / "manifest.json").is_file())
FOLDS = ROOT.parent / "forecast-validation-20261002" / "records.json"

ROWS = ("Auction history:\n\n"
        "auction_date | new/reopen | offering_$B | bid_to_cover\n"
        "2024-01-10 | reopen | 37.0 | 2.56\n"
        "2024-02-07 | new | 42.0 | 2.50\n"
        "2024-03-12 | reopen | 39.0 | 2.43\n"
        "2024-04-10 | reopen | 39.0 | --\n"
        "2024-05-08 | new | 42.0 | 2.58\n"
        "2024-11-05 | new | 42.0 | 2.61\n")
GRID = ("VINTAGE TABLE\n\n"
        "reference_month | as_of_2024-06-04 | as_of_2024-07-03 | as_of_2024-08-02 | as_of_2024-10-04\n"
        "2024-05 | 100.0 | 101.5 | 101.0 | 99.0\n"
        "2024-06 | -- | 200.0 | 202.0 | 205.0\n")


class Corpus:
    def __init__(self, texts: dict[str, str], date: str = "2024-09-30"):
        self.doc_texts = texts
        self.doc_dates = {doc_id: date for doc_id in texts}


def chunks(*doc_ids: str) -> list[Chunk]:
    return [Chunk(d, "2024-09-30", 0, 10, "x" * 10) for d in doc_ids]


TASK = {"cutoff_date": "2024-10-31", "target": {"name": "bid_to_cover_ratio", "type": "regression"}}
OWN = {"own": (frozenset({"E"}), False), "peer": (frozenset({"P"}), False), "shared": (frozenset(), True)}


def by_column(records):
    return {(r.doc_id, r.column, r.label): r for r in records}


def test_row_series_keep_exact_spans_and_skip_missing_or_future_rows():
    corpus = Corpus({"own": ROWS})
    records = by_column(build_history(TASK, {"entity_id": "E"}, corpus, OWN, chunks("own")))
    btc = records[("own", "bid_to_cover", "")]
    assert [c.period for c in btc.cells] == ["2024-01-10", "2024-02-07", "2024-03-12", "2024-05-08"]
    for cell in btc.cells:
        assert ROWS[cell.start:cell.end] == cell.raw and float(cell.raw) == cell.value
    assert btc.axis == "rows"
    assert ("own", "new/reopen", "") not in records  # nonnumeric column
    assert ("own", "auction_date", "") not in records


def test_derived_values_recompute_exactly_from_the_raw_cells():
    record = by_column(build_history(TASK, {"entity_id": "E"}, Corpus({"own": ROWS}),
                                     OWN, chunks("own")))[("own", "bid_to_cover", "")]
    values = [c.value for c in record.cells]
    assert record.last == values[-1]
    assert record.last_change == values[-1] - values[-2]
    assert record.median_window == min(6, len(values))
    assert record.median == statistics.median(values[-record.median_window:])


def test_vintage_grid_uses_only_the_entity_reference_month_across_eligible_vintages():
    corpus = Corpus({"own": GRID})
    task = dict(TASK, cutoff_date="2024-09-30")
    records = build_history(task, {"entity_id": "E", "ref_month": "2024-06"}, corpus, OWN, chunks("own"))
    assert len(records) == 1
    record = records[0]
    assert (record.axis, record.label) == ("vintages", "2024-06")
    assert [c.period for c in record.cells] == ["2024-07-03", "2024-08-02"]  # Oct 4 is after cutoff
    assert [c.value for c in record.cells] == [200.0, 202.0]
    for cell in record.cells:
        assert GRID[cell.start:cell.end] == cell.raw


@pytest.mark.parametrize("entity", [{"entity_id": "E"}, {"entity_id": "E", "a": "2024-05", "b": "2024-06"}])
def test_vintage_grid_without_one_explicit_reference_month_is_omitted(entity):
    assert build_history(TASK, entity, Corpus({"own": GRID}), OWN, chunks("own")) == []


def test_only_selected_own_or_shared_documents_are_read():
    corpus = Corpus({"own": ROWS, "peer": ROWS, "shared": ROWS, "unselected": ROWS})
    owners = dict(OWN, unselected=(frozenset({"E"}), False))
    docs = {r.doc_id for r in build_history(TASK, {"entity_id": "E"}, corpus, owners,
                                            chunks("own", "peer", "shared"))}
    assert docs == {"own", "shared"}
    assert build_history(TASK, {"entity_id": "E"}, corpus, None, chunks("own")) == []


def test_documents_dated_after_the_cutoff_are_ignored():
    corpus = Corpus({"own": ROWS}, date="2024-11-15")
    assert build_history(TASK, {"entity_id": "E"}, corpus, OWN, chunks("own")) == []


def test_columns_matching_the_target_come_first():
    records = build_history(TASK, {"entity_id": "E"}, Corpus({"own": ROWS}), OWN, chunks("own"))
    assert records[0].column == "bid_to_cover"


def test_format_is_bounded_and_omits_whole_records():
    records = build_history(TASK, {"entity_id": "E"}, Corpus({"own": ROWS}), OWN, chunks("own"))
    full = format_history(records, limit=5000)
    assert "bid_to_cover" in full and "offering_$B" in full
    short = format_history(records, limit=len(full) - 1)
    assert len(short) <= len(full) - 1 and short and full.startswith(short.rstrip("\n").rsplit("\n", 1)[0])
    assert format_history([], limit=1200) == "" and format_history(records, limit=10) == ""


def test_parsing_work_is_bounded():
    huge = "date | v\n" + "".join(f"2024-01-{1 + i % 28:02d} | {i}\n" for i in range(200_000))
    started = time.monotonic()
    build_history(TASK, {"entity_id": "E"}, Corpus({"own": huge}), OWN, chunks("own"))
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_public_units_have_exact_provenance_and_eligible_sources(unit):
    from retrieval import EvidenceIndex, build_index
    task = json.loads((unit / "task.json").read_text())
    corpus = build_index(unit / "corpus")
    owners = load_ownership(unit / "corpus")
    index = EvidenceIndex(corpus, task["cutoff_date"])
    for entity in task["entities"]:
        context = index.retrieve(task, entity, top_k=8)
        selected = {c.doc_id for c in context}
        for record in build_history(task, entity, corpus, owners, context):
            assert record.doc_id in selected and may_cite(owners, record.doc_id, entity["entity_id"])
            text = corpus.doc_texts[record.doc_id]
            for cell in record.cells:
                assert text[cell.start:cell.end] == cell.raw
                assert cell.period[:10] <= task["cutoff_date"]
        block = format_history(build_history(task, entity, corpus, owners, context), limit=1200)
        assert len(block) <= 1200


@pytest.mark.skipif(not FOLDS.is_file(), reason="Codex rolling-origin records not present")
def test_auction_records_reproduce_the_rolling_origin_benchmark():
    """Cross-implementation check against forecast-validation-20261002/records.json."""
    unit = next(u for u in UNITS if "auction" in u.name)
    from retrieval import build_index
    corpus = build_index(unit / "corpus")
    owners = load_ownership(unit / "corpus")
    folds = [r for r in json.loads(FOLDS.read_text())["records"] if r["family"] == "auction"] \
        if isinstance(json.loads(FOLDS.read_text()), dict) else \
        [r for r in json.loads(FOLDS.read_text()) if r["family"] == "auction"]
    assert len(folds) == 42
    task = json.loads((unit / "task.json").read_text())
    for fold in folds:
        doc_id = fold["entity"]
        entity = next(e for e in task["entities"] if may_cite(owners, doc_id, e["entity_id"]))
        # A fold sees the table as the prefix available at its origin, as in the benchmark.
        view = Corpus({doc_id: corpus.doc_texts[doc_id]}, date=fold["origin"])
        records = build_history(dict(task, cutoff_date=fold["origin"]), entity, view, owners,
                                [Chunk(doc_id, fold["origin"], 0, 1, "x")])
        btc = next(r for r in records if r.column == "bid_to_cover")
        assert btc.cells[-1].period == fold["history_end"]
        assert btc.last == pytest.approx(fold["predictions"]["persistence"], abs=1e-12)
        assert btc.median_window == 6
        assert btc.median == pytest.approx(fold["predictions"]["median"], abs=1e-12)


def test_blank_trailing_cell_keeps_the_row_and_its_other_columns():
    """COT tables leave the first week-over-week change blank: '... | -10.12 | '."""
    text = ("report_date | net | wow_change_net\n"
            "2024-04-30 | -143,424 | \n"
            "2024-05-07 | -31,354 | +112,070\n")
    records = by_column(build_history(TASK, {"entity_id": "E"}, Corpus({"own": text}), OWN, chunks("own")))
    assert [c.value for c in records[("own", "net", "")].cells] == [-143424.0, -31354.0]
    assert [c.raw for c in records[("own", "wow_change_net", "")].cells] == ["+112,070"]


def test_cot_public_unit_entities_get_a_block():
    unit = next(u for u in UNITS if "cotpos" in u.name)
    from retrieval import EvidenceIndex, build_index
    task = json.loads((unit / "task.json").read_text())
    corpus, owners = build_index(unit / "corpus"), load_ownership(unit / "corpus")
    index = EvidenceIndex(corpus, task["cutoff_date"])
    assert all(format_history(build_history(task, e, corpus, owners, index.retrieve(task, e, top_k=8)))
               for e in task["entities"])


SHARED = ("month | All items | Shelter | Energy | DGS10\n"
          "2024-08 | 0.19 | 0.52 | -0.78 | 3.81\n"
          "2024-09 | 0.18 | 0.22 | -1.85 | 3.73\n")


@pytest.mark.parametrize("entity, first", [
    ({"entity_id": "CPI_ENERGY", "name": "Energy"}, "Energy"),
    ({"entity_id": "UST10Y", "name": "10-Year Treasury yield", "series_fred": "DGS10"}, "DGS10"),
])
def test_the_entity_own_series_leads_a_shared_table(entity, first):
    owners = {"shared": (frozenset(), True)}
    records = build_history(dict(TASK, target={"name": "zeta"}), entity, Corpus({"shared": SHARED}),
                            owners, chunks("shared"))
    assert records[0].column == first


def test_large_values_are_printed_without_exponents():
    text = "date | open_interest\n" + "".join(f"2024-0{m}-01 | {1_400_000 + m * 10_001:,}\n" for m in range(1, 8))
    block = format_history(build_history(TASK, {"entity_id": "E"}, Corpus({"own": text}), OWN, chunks("own")))
    assert "e+" not in block and "median of last 6 = 1445004.5" in block
