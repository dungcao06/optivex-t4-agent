"""Regression checks for bounded, attributable evidence retrieval."""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "track4-analysis-public"))


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
