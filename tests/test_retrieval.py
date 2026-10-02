"""Regression checks for bounded, attributable evidence retrieval."""
from __future__ import annotations

import importlib
import importlib.util
from itertools import islice
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))
sys.path.insert(0, str(UPSTREAM))


def retrieval_module():
    # Before implementation this is an assertion failure, not a collection error.
    assert importlib.util.find_spec("retrieval") is not None, "bounded retrieval is missing"
    return importlib.import_module("retrieval")


def write_doc(root: Path, doc_id: str, text: str, **metadata) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"doc_id": doc_id, "doc_date": "2024-01-31", "text": text}
    payload.update(metadata)
    (root / f"{doc_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_large_flat_and_span_documents_keep_exact_offsets(tmp_path):
    """Catches chunk offsets being measured against normalized or chopped text."""
    mod = retrieval_module()
    flat = "Opening — naïve résumé.\n\n" + ("EPS guidance rose 12 percent this quarter. " * 200)
    write_doc(tmp_path, "flat", flat)
    spans = ["First published section. " * 130, "", "Second section — net income fell. " * 100]
    (tmp_path / "spans.json").write_text(json.dumps({
        "doc_id": "spans", "doc_date": "2024-01-30",
        "spans": [{"text": text} for text in spans],
    }), encoding="utf-8")
    corpus = mod.build_index(tmp_path)
    assert corpus.doc_texts == {"flat": flat, "spans": " ".join(spans)}
    assert len(corpus.chunks) > 4
    for chunk in corpus.chunks:
        assert chunk.text == corpus.doc_texts[chunk.doc_id][chunk.span_start:chunk.span_end]
        assert 0 < len(chunk.text) <= 2200
    for doc_id, text in corpus.doc_texts.items():
        pieces = [chunk for chunk in corpus.chunks if chunk.doc_id == doc_id]
        assert pieces[0].span_start == 0
        assert pieces[-1].span_end == len(text)
        assert all(b.span_start <= a.span_end for a, b in zip(pieces, pieces[1:]))


@pytest.mark.parametrize("doc_date", [None, "", "2024-02-30", "2024-1-01", "not-a-date", "2024-02-02"])
def test_missing_invalid_and_future_dates_never_reach_retrieval(tmp_path, doc_date):
    """Catches lexical-date filtering admitting malformed or post-cutoff evidence."""
    mod = retrieval_module()
    write_doc(tmp_path, "valid", "Alpha credit liquidity outlook is stable.")
    write_doc(tmp_path, "invalid", "Alpha credit liquidity outlook is stable.", doc_date=doc_date)
    corpus = mod.build_index(tmp_path)
    chunks = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": "credit_event"}, "prompt": "Assess credit liquidity."},
        {"entity_id": "Alpha"},
    )
    assert [chunk.doc_id for chunk in chunks] == ["valid"]


def test_multimegabyte_filings_stay_within_evidence_budget(tmp_path):
    """Catches full filings being injected into one per-entity prompt."""
    mod = retrieval_module()
    filing = "\n\n".join(
        f"Quarter {n}: Alpha EPS guidance revenue profit margin. " + "financial discussion " * 110
        for n in range(1100)
    )
    assert len(filing) > 2_000_000
    write_doc(tmp_path, "alpha", filing, cik="0000000010")
    corpus = mod.build_index(tmp_path)
    chunks = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": "eps_growth"}, "prompt": "Forecast EPS growth."},
        {"entity_id": "ALPHA", "cik": "0000000010"},
        top_k=100,
    )
    assert chunks
    assert sum(len(chunk.text) for chunk in chunks) <= 16000
    assert all(len(chunk.text) <= 2200 for chunk in chunks)
    assert len({(chunk.doc_id, chunk.span_start, chunk.span_end) for chunk in chunks}) == len(chunks)


def test_target_terms_retrieve_relevant_passage_not_generic_earnings(tmp_path):
    """Catches a fixed earnings query dropping the actual non-EPS target."""
    mod = retrieval_module()
    write_doc(tmp_path, "revenue", "Alpha posted revenue growth and earnings results.")
    write_doc(tmp_path, "solvency", "Alpha has tight liquidity, debt maturity, covenant default and bankruptcy risk.")
    index = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31")
    chunks = index.retrieve(
        {"target": {"name": "credit_event_12m"}, "prompt": "Assess liquidity, debt maturity, covenant default and bankruptcy risk."},
        {"entity_id": "ALPHA"}, top_k=1,
    )
    assert chunks[0].doc_id == "solvency"


@pytest.mark.parametrize("doc_ids", [("archive_a", "record_z"), ("record_z", "archive_a")])
@pytest.mark.parametrize("tenor, expected_index", [("2-Year", 0), ("10-Year", 1)])
def test_numeric_identity_retrieval_survives_document_renaming(tmp_path, doc_ids, tenor, expected_index):
    """Catches tenor identity being discarded so names or document order pick the source."""
    mod = retrieval_module()
    for doc_id, maturity in zip(doc_ids, ("2-Year", "10-Year")):
        write_doc(tmp_path, doc_id, f"Treasury {maturity} Note futures positioning history.")
    corpus = mod.build_index(tmp_path)
    selected = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": "positioning_change"}},
        {"name": f"Treasury {tenor} Note futures"}, top_k=1,
    )
    assert [chunk.doc_id for chunk in selected] == [doc_ids[expected_index]]
    assert selected[0].text == corpus.doc_texts[selected[0].doc_id]


@pytest.mark.parametrize("identity", [
    "7-Year", "7\u2010Year", "7\u2011Year", "7\u2012Year", "7\u2013Year", "7\u2014Year",
    "7\u2015Year", "7\u2212Year", "7 - Year", "7\t-\tYear", "7\u00a0-\u00a0Year", "7\u202f-\u202fYear",
])
def test_numeric_identity_retrieval_matches_hyphens_whitespace_and_case(tmp_path, identity):
    """Catches spelling variants dropping the joined identity from index or query."""
    mod = retrieval_module()
    text = f"Treasury {identity} Note futures positioning history."
    write_doc(tmp_path, "a_unrelated", text.replace("7", "3"))
    write_doc(tmp_path, "z_relevant", text)
    corpus = mod.build_index(tmp_path)
    selected = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": "positioning_change"}},
        {"name": "TREASURY 7-yEaR NOTE FUTURES"}, top_k=1,
    )
    assert [chunk.doc_id for chunk in selected] == ["z_relevant"]
    assert selected[0].text == text
    assert (selected[0].span_start, selected[0].span_end) == (0, len(text))


@pytest.mark.parametrize("unit", ["year", "month", "day"])
@pytest.mark.parametrize("requested, other", [("5", "0.5"), ("0.5", "5")])
def test_numeric_identity_retrieval_keeps_decimal_magnitude(tmp_path, unit, requested, other):
    """Catches fractional tails such as 0.5-year falsely becoming the 5-year identity."""
    mod = retrieval_module()
    write_doc(tmp_path, "a_other", f"Treasury {other}-{unit} yield history.")
    write_doc(tmp_path, "z_requested", f"Treasury {requested}-{unit} yield history.")
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": "yield_change"}}, {"tenor": f"{requested}-{unit}"}, top_k=1,
    )
    assert [chunk.doc_id for chunk in selected] == ["z_requested"]


@pytest.mark.parametrize("prefix", ["Treasury 2-Year, ", "Treasury 2-Year,", "Maturity. ", "Maturity.\t"])
def test_numeric_identity_retrieval_preserves_lists_and_sentence_boundaries(tmp_path, prefix):
    """Catches prose punctuation hiding a requested maturity in a list or sentence."""
    mod = retrieval_module()
    write_doc(tmp_path, "a_other", f"{prefix}3-Year Note futures positioning history.")
    text = f"{prefix}10-Year Note futures positioning history."
    write_doc(tmp_path, "z_requested", text)
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": "positioning_change"}}, {"name": "Treasury 10-Year Note futures"}, top_k=1,
    )
    assert [chunk.doc_id for chunk in selected] == ["z_requested"]
    assert selected[0].text == text


def test_numeric_identity_tokens_retain_existing_terms_without_decimal_tail_aliases():
    """Catches compound terms replacing word tokens or starting inside a decimal."""
    tokens = retrieval_module()._tokens("Treasury 0.5-year and .5-year versus 5-year, 10-Year, 7-month.")
    assert tokens.count("year") == 4
    assert {"treasury", "versus", "10", "0.5year", "5year", "10year", "7month"} <= set(tokens)
    assert tokens.count("5year") == 1


@pytest.mark.parametrize("text, alias", [
    (".5-year", "5year"), ("1.2.5-year", "5year"), ("ref35-year", "35year"),
    ("35 year", "35year"), ("35\tyear", "35year"), ("35\u00a0year", "35year"),
    ("0,5-year", "5year"), ("1,000-year", "000year"), ("1,000.5-year", "000.5year"),
    ("-5-year", "5year"), ("+5-year", "5year"), ("-0.5-year", "0.5year"),
    ("2\u20135-year", "5year"), ("1e-5-year", "5year"),
    ("- 5-year", "5year"), ("2\u2013 5-year", "5year"), ("0, 5-year", "5year"),
    ("+\t\u00a0 5-year", "5year"), ("1.\u202f5-year", "5year"),
])
def test_numeric_identity_tokens_do_not_join_partial_numbers_or_ordinary_prose(text, alias):
    """Catches identities manufactured inside numbers or from ordinary numeric prose."""
    tokens = retrieval_module()._tokens(text)
    assert alias not in tokens


@pytest.mark.parametrize("separator", [
    "\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029",
])
@pytest.mark.parametrize("parts", [("35", "-year"), ("35-", "year")])
def test_numeric_identity_tokens_do_not_join_across_line_boundaries(separator, parts):
    """Catches Unicode/control line breaks being mistaken for horizontal spacing."""
    text = separator.join(parts)
    assert "35year" not in retrieval_module()._tokens(text)


@pytest.mark.parametrize("form, alias", [("10-Q", "10q"), ("10-K", "10k")])
def test_numeric_identity_tokens_do_not_join_single_letter_document_forms(form, alias):
    """Catches new filing-type matches overpowering predictive terms in queries."""
    tokens = retrieval_module()._tokens(form)
    assert "10" in tokens
    assert alias not in tokens


def test_numeric_identity_retrieval_does_not_promote_calendar_prose_on_filing_cover():
    """Catches bare numeric prose creating identity matches on a filing cover page."""
    mod = retrieval_module()
    unit = UPSTREAM / "units/t4-credit-event-2023"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    entity = next(entity for entity in task["entities"] if entity["entity_id"] == "ODFL")
    selected = mod.EvidenceIndex(mod.build_index(unit / "corpus"), task["cutoff_date"]).retrieve(task, entity)
    assert not any(chunk.doc_id == "EDGAR_0000878927_10K_20230222" and chunk.span_start == 0
                   for chunk in selected)


def test_numeric_identity_retrieval_does_not_promote_document_form_on_filing_cover():
    """Catches single-letter form terms returning a cover page for an earnings target."""
    mod = retrieval_module()
    unit = UPSTREAM / "units/t4-eps-yoy-2023Q2-mixed"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    entity = next(entity for entity in task["entities"] if entity["entity_id"] == "HON")
    selected = mod.EvidenceIndex(mod.build_index(unit / "corpus"), task["cutoff_date"]).retrieve(task, entity)
    assert not any(chunk.doc_id == "EDGAR_0000773840_10Q_20230427" and chunk.span_start == 0
                   for chunk in selected)


def test_public_cot_numeric_identity_retrieves_matching_later_history():
    """Catches 2-Year retrieval spending its last slot on the 10-Year history tail."""
    mod = retrieval_module()
    unit = UPSTREAM / "units/t4-cotpos-202411-us10"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = mod.build_index(unit / "corpus")
    index = mod.EvidenceIndex(corpus, task["cutoff_date"])
    for entity_id, expected, other in [
        ("UST_2Y", ("COT_UST_2Y_20241025", 1981, 3270), ("COT_UST_10Y_20241025", 1933, 3213)),
        ("UST_10Y", ("COT_UST_10Y_20241025", 1933, 3213), ("COT_UST_2Y_20241025", 1981, 3270)),
    ]:
        entity = next(entity for entity in task["entities"] if entity["entity_id"] == entity_id)
        selected = index.retrieve(task, entity)
        spans = {(chunk.doc_id, chunk.span_start, chunk.span_end) for chunk in selected}
        assert expected in spans, entity_id
        assert other not in spans, entity_id
        assert sum(len(chunk.text) for chunk in selected) <= 16000
        assert all(chunk.text == corpus.doc_texts[chunk.doc_id][chunk.span_start:chunk.span_end]
                   and len(chunk.text) <= 2200 for chunk in selected)


def test_cik_affinity_separates_companies_but_preserves_shared_macro(tmp_path):
    """Catches retrieval spending the budget on another issuer's filing."""
    mod = retrieval_module()
    write_doc(tmp_path, "issuer_a", "Net interest income and earnings guidance for the bank.", cik="0000000010", ticker="AAA")
    write_doc(tmp_path, "issuer_b", "Net interest income and earnings guidance for the bank.", cik="0000000020", ticker="BBB")
    write_doc(tmp_path, "macro", "Interest rates and inflation influence bank earnings guidance.", source="FRED")
    index = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31")
    task = {"target": {"name": "eps_growth"}, "prompt": "Forecast bank earnings guidance from net interest income and interest rates."}
    a = index.retrieve(task, {"entity_id": "AAA", "cik": "0000000010", "corpus_ref": "corpus/"})
    b = index.retrieve(task, {"entity_id": "BBB", "cik": "0000000020", "corpus_ref": "corpus/"})
    assert {chunk.doc_id for chunk in a} == {"issuer_a", "macro"}
    assert {chunk.doc_id for chunk in b} == {"issuer_b", "macro"}
    assert a[0].doc_id == "issuer_a"
    assert b[0].doc_id == "issuer_b"


def test_specific_corpus_reference_guides_entity_evidence(tmp_path):
    """Catches ignoring an entity's declared evidence subtree."""
    mod = retrieval_module()
    write_doc(tmp_path / "a", "doc_a", "Bank earnings outlook improved.")
    write_doc(tmp_path / "b", "doc_b", "Bank earnings outlook deteriorated.")
    write_doc(tmp_path, "shared", "Interest rate outlook matters for bank earnings.")
    index = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31")
    chunks = index.retrieve(
        {"target": {"name": "eps"}, "prompt": "Predict bank earnings outlook."},
        {"entity_id": "A", "corpus_ref": "corpus/a/"},
    )
    assert chunks[0].doc_id == "doc_a"
    assert "shared" in {chunk.doc_id for chunk in chunks}
    assert "doc_b" not in {chunk.doc_id for chunk in chunks}


def test_retrieval_covers_two_supporting_documents_without_duplicate_overlap(tmp_path):
    """Catches many near-identical windows displacing independent support."""
    mod = retrieval_module()
    text = "Alpha earnings guidance liquidity and revenue outlook. " * 300
    write_doc(tmp_path, "quarterly", text, cik="0000000010")
    write_doc(tmp_path, "recent", "Alpha earnings guidance changed after a new financing agreement.", cik="0000000010")
    index = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31")
    chunks = index.retrieve(
        {"target": {"name": "eps"}, "prompt": "Predict earnings guidance."},
        {"entity_id": "ALPHA", "cik": "0000000010"}, top_k=4,
    )
    assert {chunk.doc_id for chunk in chunks} == {"quarterly", "recent"}
    assert len({chunk.text for chunk in chunks}) == len(chunks)
    for left, right in zip(chunks, chunks[1:]):
        if left.doc_id == right.doc_id:
            overlap = max(0, min(left.span_end, right.span_end) - max(left.span_start, right.span_start))
            assert overlap <= min(len(left.text), len(right.text)) // 3


def test_known_issuer_identity_does_not_promote_cover_page(tmp_path):
    """Catches identifiers overpowering prediction evidence within one issuer."""
    mod = retrieval_module()
    write_doc(tmp_path, "cover", "Alpha Provincial Financial Corporation. APFC. CIK 0000000010. Information Technology. USD. Three months ended 2024-06-30. Securities Exchange Commission Form 10-Q.", cik="0000000010")
    write_doc(tmp_path, "results", "Diluted EPS grew as net interest income and fee revenue improved.", cik="0000000010")
    index = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31")
    chunks = index.retrieve(
        {"target": {"name": "eps_growth"}, "prompt": "Predict diluted EPS from net interest income and fee revenue."},
        {"entity_id": "APFC", "name": "Alpha Provincial Financial Corporation", "cik": "0000000010", "sector": "Information Technology", "currency": "USD", "quarter_reported": "Three months ended 2024-06-30"}, top_k=1,
    )
    assert chunks[0].doc_id == "results"


@pytest.mark.parametrize("duplicates", [False, True])
def test_large_retrieval_does_not_repeatedly_rescan_rejected_candidates(duplicates):
    """Counts candidate identity reads, catching quadratic selection work."""
    mod = retrieval_module()
    from baselines.strong_rag_baseline.indexer import Chunk, IndexedCorpus

    class CountedChunk(Chunk):
        identity_reads = 0

        def __getattribute__(self, name):
            if name == "doc_id":
                type(self).identity_reads += 1
            return super().__getattribute__(name)

    chunks = []
    for i in range(1200):
        text = "earnings metric " + " ".join(f"data{0 if duplicates else i}n{j}" for j in range(100))
        text = (text * 4)[:2200]
        chunks.append(CountedChunk("issuer", "2024-01-31", i * 2200, (i + 1) * 2200, text))
    corpus = IndexedCorpus(chunks, {"issuer": "".join(chunk.text for chunk in chunks)}, {"issuer": "2024-01-31"})
    index = mod.EvidenceIndex(corpus, "2024-01-31")
    CountedChunk.identity_reads = 0
    selected = index.retrieve({"target": {"name": "earnings"}}, {"entity_id": "A"})
    assert selected
    assert sum(len(chunk.text) for chunk in selected) <= 16000
    assert CountedChunk.identity_reads < 80 * len(chunks)


def test_unfamiliar_vocabulary_still_provides_bounded_eligible_evidence(tmp_path):
    """Catches a zero-lexical-match task failing solely because retrieval is empty."""
    mod = retrieval_module()
    write_doc(tmp_path, "old", "Solvency remained sound and cash increased.", doc_date="2023-12-01")
    write_doc(tmp_path, "new", "Obligations were repaid ahead of schedule.", doc_date="2024-01-30")
    write_doc(tmp_path, "future", "Obligations fell sharply.", doc_date="2024-02-10")
    index = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31")
    selected = index.retrieve({"target": {"name": "credit_risk"}}, {"entity_id": "UNKNOWN"}, top_k=1)
    assert [chunk.doc_id for chunk in selected] == ["new"]


@pytest.mark.parametrize("markdown", [False, True])
def test_compact_table_crossing_window_keeps_header_units_and_last_row(tmp_path, markdown):
    """Catches a passage boundary separating a series header from its recent rows."""
    mod = retrieval_module()
    header = "date | ALPHA | BETA\n"
    table = header + ("--- | ---: | ---:\n" if markdown else "")
    table += "".join(f"2030-01-{day:02d} | 4.21 | 3.75\n" for day in range(1, 25))
    caption = "Rates — units: percent:\n"
    prefix = "Résumé and context. " * 85 + "\n\n"
    text = prefix + caption + table + "\nFollowing commentary. " * 60
    write_doc(tmp_path, "generic", text)
    corpus = mod.build_index(tmp_path)
    chunks = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": "yield_change"}}, {"entity_id": "ALPHA"},
    )
    assert any(caption + table in chunk.text for chunk in chunks)
    for chunk in corpus.chunks:
        assert chunk.text == text[chunk.span_start:chunk.span_end]
        assert 0 < len(chunk.text) <= 2200
        assert not len(prefix) < chunk.span_start < len(prefix + caption + table)
        assert not len(prefix) < chunk.span_end < len(prefix + caption + table)
    assert corpus.chunks[0].span_start == 0
    assert corpus.chunks[-1].span_end == len(text)
    assert all(b.span_start <= a.span_end for a, b in zip(corpus.chunks, corpus.chunks[1:]))


def test_public_rates_retrieval_preserves_complete_table_for_every_maturity():
    """Catches table-tail loss through actual ranked retrieval, not only indexing."""
    mod = retrieval_module()
    unit = UPSTREAM / "units/t4-fomc-curve-20240918"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = mod.build_index(unit / "corpus")
    text = corpus.doc_texts["RATES_SNAPSHOT_20240919"]
    full_table = text[text.index("U.S. Treasury constant-maturity yields, percent, recent closes:"):]
    assert "2024-09-19 | 3.93 | 3.59" in full_table
    assert len(task["entities"]) == 6
    index = mod.EvidenceIndex(corpus, task["cutoff_date"])
    for entity in task["entities"]:
        selected = index.retrieve(task, entity)
        assert any(full_table in chunk.text for chunk in selected), entity["entity_id"]
        assert sum(len(chunk.text) for chunk in selected) <= 16000
        assert all(chunk.text == corpus.doc_texts[chunk.doc_id][chunk.span_start:chunk.span_end]
                   and len(chunk.text) <= 2200 for chunk in selected)


@pytest.mark.parametrize("prefix_length", [0, 100, 1500, 2000, 2199, 4000])
def test_multiple_unicode_tables_have_no_internal_boundaries_or_coverage_gaps(prefix_length):
    """Catches overlap starts inside tables and non-progress at adjacent boundaries."""
    mod = retrieval_module()
    table_a = "| Période | Valeur |\r\n| :--- | ---: |\r\n" + "| février | 4.2 |\r\n" * 40
    table_b = "Date | RATE\n" + "2030-01-01 | 3.1\n" * 65
    prefix = "x" * prefix_length + "\n\n"
    bridge = "\nOrdinary interstitial discussion.\n\n"
    text = prefix + table_a + bridge + table_b + "\n" + "z" * 2600
    intervals = [(len(prefix), len(prefix + table_a)),
                 (len(prefix + table_a + bridge), len(prefix + table_a + bridge + table_b))]
    windows = list(islice(mod._passages(text), 100))
    assert len(windows) < 100
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    for start, end in windows:
        assert 0 < end - start <= 2200
        for table_start, table_end in intervals:
            assert not table_start < start < table_end
            assert not table_start < end < table_end
    assert all(a[0] < b[0] <= a[1] for a, b in zip(windows, windows[1:]))
    assert all(any(start <= left and end >= right for start, end in windows)
               for left, right in intervals)


def test_exact_limit_table_is_whole_without_oversized_caption():
    """Catches dropping a fitting table because its adjacent units exceed the cap."""
    mod = retrieval_module()
    table = "Name | Value\nALPHA | " + "1" * 2178 + "\n"
    assert len(table) == 2200
    text = "Context. " * 180 + "\nUnits: percent\n" + table + "Tail. " * 500
    pieces = [text[start:end] for start, end in mod._passages(text)]
    assert table in pieces
    assert all(len(piece) <= 2200 for piece in pieces)


@pytest.mark.parametrize("table", [
    "date | RATE\n" + "2030-01-01 | 3.1\n" * 200,  # Oversized whole block.
    "date | RATE\n2030-01-01 | 3.1 | extra\n" + "2030-01-02 | 3.2\n" * 50,
    "date | RATE\n--- | ---\n--- | ---\n",  # No data.
    "2029 | 2030\n1 | 2\n",  # No recognizable header.
    "Name \\| escaped | Value\n" + "ALPHA | 3.1\n" * 65,
])
def test_oversized_or_unrecognized_table_retains_bounded_gap_free_windows(table):
    """Catches malformed/oversized blocks expanding windows or stalling traversal."""
    mod = retrieval_module()
    text = "Context. " * 180 + "\n" + table + "\n" + "Later context. " * 300
    windows = list(islice(mod._passages(text), 100))
    assert len(windows) < 100
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    assert all(0 < end - start <= 2200 for start, end in windows)
    assert all(a[0] < b[0] <= a[1] for a, b in zip(windows, windows[1:]))


def test_many_small_tables_traverse_with_bounded_number_of_windows():
    """Catches repeated one-character progress around dense protected blocks."""
    mod = retrieval_module()
    block = "Name | Value\nALPHA | 3.1\nBETA | 2.1\n\n"
    text = block * 20000
    windows = list(islice(mod._passages(text), 1000))
    assert len(windows) < 1000
    assert windows[-1][1] == len(text)
    assert all(0 < end - start <= 2200 for start, end in windows)
    assert all(a[0] < b[0] <= a[1] for a, b in zip(windows, windows[1:]))


@pytest.mark.parametrize("table", [
    "date | RATE\n2030-01-01 | 3.1 | extra\n" + "2030-01-02 | 3.2\n" * 50,
    "2029 | 2030\n" + "1 | 2\n" * 130,
    "Name \\| escaped | Value\n" + "ALPHA | 3.1\n" * 65,
    "Name | Value\n--- | ---\n" + "--- | ---\n" * 80,
])
def test_malformed_pipe_runs_do_not_trigger_table_preservation(table):
    """Catches false positives silently changing evidence for inconsistent pipe prose."""
    mod = retrieval_module()
    text = "Context. " * 180 + "\n" + table + "\n" + "Later context. " * 300
    pieces = [text[start:end] for start, end in mod._passages(text)]
    assert not any(table in piece for piece in pieces)


def test_plain_text_keeps_existing_overlap_windows():
    """Catches a table-only fix altering ordinary hard-window traversal."""
    assert list(retrieval_module()._passages("x" * 5000)) == [(0, 2200), (2000, 4200), (4000, 5000)]


@pytest.mark.parametrize("metric", ["EPS growth", "change in diluted earnings per share"])
@pytest.mark.parametrize("baseline", [
    "Consolidated income statement. Earnings per share—basic $8.41 $7.32. "
    "Earnings per share—assuming dilution $8.39 $7.30.",
    "Net income of $1.7 billion, or $8.39 per diluted common share, "
    "for the quarter, compared to $7.30 per diluted common share.",
    "Earnings/(loss) per share of common stock: Assuming dilution: "
    "Continuing operations $8.40 $7.32 Discontinued operations (0.01) (0.02) "
    "Total $8.39 $7.30 Basic: Total $8.41 $7.32.",
])
def test_numeric_complement_preserves_drivers_and_eligible_exact_spans(tmp_path, metric, baseline):
    """Catches lexical driver matches crowding out an issuer's absolute EPS."""
    mod = retrieval_module()
    for i in range(10):
        write_doc(tmp_path, f"driver{i}",
                  f"Outlook guidance margins revenue credit provisions {metric}. "
                  + f"Operational factor{i} improved. " * 70, cik="55")
    write_doc(tmp_path, "baseline", "Résumé — " + baseline, cik="55")
    write_doc(tmp_path, "future", baseline, cik="55", doc_date="2024-02-01")
    write_doc(tmp_path, "peer", baseline, cik="66")
    write_doc(tmp_path, "cover", "Earnings Per Share 74. Securities registration 2024.", cik="55")
    corpus = mod.build_index(tmp_path)
    selected = mod.EvidenceIndex(corpus, "2024-01-31").retrieve(
        {"target": {"name": "unseen_measure"}, "prompt": f"Forecast {metric} from outlook guidance margins revenue credit provisions."},
        {"entity_id": "RENAMED", "cik": "55"}, top_k=4,
    )
    assert "baseline" in {c.doc_id for c in selected}
    assert sum(c.doc_id.startswith("driver") for c in selected) == 3
    assert not {"future", "peer", "cover"} & {c.doc_id for c in selected}
    assert len(selected) <= 4 and sum(len(c.text) for c in selected) <= 16000
    assert all(c.text == corpus.doc_texts[c.doc_id][c.span_start:c.span_end] for c in selected)


@pytest.mark.parametrize("unit_name, entity_id, anchor, driver_start", [
    ("t4-eps-yoy-2023Q2-mixed", "HON", "2.07", 91841),
    ("t4-eps-yoy-2023Q2-mixed", "IBM", "1.01", 129188),
    ("t4-eps-growth-2024Q3-banks", "PNC", "3.39", 27958),
])
def test_public_eps_complement_restores_absolute_baselines(unit_name, entity_id, anchor, driver_start):
    """Catches the three audited baseline omissions while retaining useful drivers."""
    mod = retrieval_module()
    unit = UPSTREAM / "units" / unit_name
    task = json.loads((unit / "task.json").read_text())
    entity = next(e for e in task["entities"] if e["entity_id"] == entity_id)
    corpus = mod.build_index(unit / "corpus")
    selected = mod.EvidenceIndex(corpus, task["cutoff_date"]).retrieve(task, entity)
    assert any(anchor in c.text and ("Earnings" in c.text or "per diluted common share" in c.text
                                    or "Consolidated earnings" in c.text) for c in selected)
    assert any(c.span_start == driver_start for c in selected)
    assert len(selected) <= 8 and sum(len(c.text) for c in selected) <= 16000
    assert all(len(c.text) <= 2200 and c.text == corpus.doc_texts[c.doc_id][c.span_start:c.span_end]
               for c in selected)


@pytest.mark.parametrize("top_k", [0, 1, 2, 8])
def test_complement_respects_small_slot_limits_and_avoids_duplicate_baselines(tmp_path, top_k):
    """Catches a reserved baseline exceeding top_k or evicting every driver."""
    mod = retrieval_module()
    write_doc(tmp_path, "driver", "EPS outlook guidance revenue margins " * 40, cik="55")
    write_doc(tmp_path, "baseline", "Earnings per share basic $4.22 diluted $4.20.", cik="55")
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": "eps"}, "prompt": "EPS outlook guidance revenue margins"},
        {"cik": "55"}, top_k=top_k,
    )
    assert len(selected) <= top_k
    assert len({(c.doc_id, c.span_start, c.span_end) for c in selected}) == len(selected)
    if top_k:
        assert selected[0].doc_id == "driver"
    if top_k > 1:
        assert {c.doc_id for c in selected} == {"driver", "baseline"}


@pytest.mark.parametrize("target", ["yield_change", "credit_event"])
def test_eps_baseline_does_not_displace_unrelated_target_evidence(tmp_path, target):
    """Catches unconditional numeric expansion changing non-EPS tasks."""
    mod = retrieval_module()
    for i in range(3):
        write_doc(tmp_path, f"driver{i}", f"{target} liquidity outlook factor{i} " * 30, cik="55")
    write_doc(tmp_path, "baseline", "Earnings per share basic $4.22 diluted $4.20.", cik="55")
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": target}}, {"cik": "55"}, top_k=2,
    )
    assert all(c.doc_id.startswith("driver") for c in selected)


@pytest.mark.parametrize("decoy", [
    "Adjusted earnings per share basic $9.30 diluted $9.20.",
    "Earnings per share assuming dilution increased due to segment profit, "
    "which impacted earnings per share by $0.40. Total impact $0.50.",
    "Earnings Per Share 74. Consolidated statements 2024. Basic and diluted.",
])
def test_complement_does_not_reserve_slot_for_adjustments_deltas_or_contents(tmp_path, decoy):
    """Catches treating EPS drivers or filing page numbers as absolute baselines."""
    mod = retrieval_module()
    for i in range(3):
        write_doc(tmp_path, f"driver{i}", f"EPS outlook guidance revenue margins factor{i} " * 30, cik="55")
    write_doc(tmp_path, "decoy", decoy, cik="55")
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": "eps"}, "prompt": "EPS outlook guidance revenue margins"},
        {"cik": "55"}, top_k=2,
    )
    assert all(c.doc_id.startswith("driver") for c in selected)


def test_existing_absolute_eps_evidence_does_not_cost_an_extra_driver_slot(tmp_path):
    """Catches adding a redundant baseline when the driver ranking already has one."""
    mod = retrieval_module()
    write_doc(tmp_path, "selected", "EPS outlook guidance revenue margins. "
              "Net income $1.2 billion or $4.20 per diluted common share.", cik="55")
    write_doc(tmp_path, "driver", "EPS outlook guidance revenue margins improved.", cik="55")
    write_doc(tmp_path, "redundant", "Earnings per share basic $4.22 diluted $4.20 Total $4.20.", cik="55")
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": "eps"}, "prompt": "EPS outlook guidance revenue margins"},
        {"cik": "55"}, top_k=2,
    )
    assert {c.doc_id for c in selected} == {"selected", "driver"}


@pytest.mark.parametrize("baseline", [
    "GAAP diluted EPS was $8.39 versus $7.30 in the prior quarter.",
    "Diluted earnings per share $8.39 $7.30.",
])
def test_explicit_diluted_eps_levels_need_no_basic_share_row(tmp_path, baseline):
    """Catches missing standalone reported levels without a basic/diluted table."""
    mod = retrieval_module()
    for i in range(3):
        write_doc(tmp_path, f"driver{i}", f"EPS outlook revenue guidance factor{i}. " * 35, cik="55")
    write_doc(tmp_path, "baseline", baseline, cik="55")
    selected = mod.EvidenceIndex(mod.build_index(tmp_path), "2024-01-31").retrieve(
        {"target": {"name": "eps"}, "prompt": "EPS outlook revenue guidance"},
        {"cik": "55"}, top_k=2,
    )
    assert selected[0].doc_id.startswith("driver")
    assert selected[1].doc_id == "baseline"
