"""First-pass reasons checked against pinned 5.2.2 rules and toolkit schema."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public-ede7381"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UPSTREAM))

from baselines.guardrails_example import citation_rail as rail
from baselines.strong_rag_baseline.indexer import Chunk
from reasons import build_reasons

QUOTE = "Demand increased by 5 percent in the reported quarter."
MECHANISM = "Continuing demand supports a modest increase in the forecast."


def fixture(count=1, *, classification=False, quote=QUOTE, start=41):
    ids = [f"entity-{i}" for i in range(count)]
    task = {"task_id": "test", "cutoff_date": "2024-06-30", "interval_level": 0.9,
            "entities": [{"entity_id": eid} for eid in ids],
            "target": {"type": "classification" if classification else "regression",
                       "labels": ["up", "flat", "down"]}}
    predictions, raw, contexts, corpus = [], {}, [], {}
    for i, eid in enumerate(ids):
        row = {"entity_id": eid, "interval": {"level": 0.9, "lo": -1., "hi": 4.},
               "claims": [{"doc_id": f"doc-{i}", "span_start": start,
                           "span_end": start + len(quote), "claim": quote}]}
        row.update({"label": "up"} if classification else {"point_forecast": 2.5})
        predictions.append(row)
        implication = f"{eid}: label=up" if classification else f"{eid}: point_forecast=2.5"
        raw[eid] = {**copy.deepcopy(row), "reason": {"premise": quote, "mechanism": MECHANISM,
                    "answer_implication": implication, "doc_id": f"doc-{i}", "quote": quote}}
        contexts.append([Chunk(f"doc-{i}", "2024-06-01", start, start + len(quote), quote)])
        corpus[f"doc-{i}"] = rail.CorpusDoc(f"doc-{i}", " " * start + quote, "2024-06-01")
    return task, predictions, raw, contexts, corpus


def validate(task, predictions, reasons, corpus):
    from importlib.resources import files
    import jsonschema
    schema = json.loads(files("qfbench2_common").joinpath("schemas/analysis.schema.json").read_text())
    answer = {"schema_version": "3", "task_id": task["task_id"], "entity_predictions": predictions}
    if reasons:
        answer["submitted_reasons"] = reasons
    jsonschema.validate(answer, schema)
    assert rail.check_submitted_reasons(answer, corpus, task["cutoff_date"]) == []
    assert all(p["judged"] for p in rail.reasons_judged(answer, corpus, task["cutoff_date"]))


@pytest.mark.parametrize("classification", [False, True])
def test_valid_reason_preserves_model_text_and_original_offsets(classification):
    task, predictions, raw, contexts, corpus = fixture(classification=classification)
    before = copy.deepcopy((task, predictions, raw, contexts))
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == 1
    reason = reasons[0]
    assert reason["mechanism"] == MECHANISM
    assert reason["premise"] == QUOTE
    assert reason["answer_implication"] == raw["entity-0"]["reason"]["answer_implication"]
    assert reason["scope"] == {"entities": ["entity-0"]}
    citation = reason["citations"][0]
    assert citation == {"doc_id": "doc-0", "span_start": 41, "span_end": 41 + len(QUOTE)}
    assert (task, predictions, raw, contexts) == before
    validate(task, predictions, reasons, corpus)


def test_mapping_contexts_shuffled_predictions_and_three_reason_limit():
    task, predictions, raw, contexts, corpus = fixture(5)
    contexts = {e["entity_id"]: c for e, c in zip(task["entities"], contexts)}
    reasons = build_reasons(task, list(reversed(predictions)), raw, contexts)
    assert [r["reason_id"] for r in reasons] == ["r1", "r2", "r3"]
    assert [r["scope"]["entities"] for r in reasons] == [[f"entity-{i}"] for i in range(3)]
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("reason", [None, {}, [], "text", 3, {"premise": QUOTE}])
def test_missing_or_malformed_reason_is_omitted(reason):
    task, predictions, raw, contexts, corpus = fixture()
    raw["entity-0"]["reason"] = reason
    result = build_reasons(task, predictions, raw, contexts)
    assert result == []
    validate(task, predictions, result, corpus)


@pytest.mark.parametrize("field", ["premise", "mechanism", "answer_implication", "doc_id", "quote"])
@pytest.mark.parametrize("value", [None, [], 1, "", "   "])
def test_invalid_model_fields_never_invent_a_replacement(field, value):
    task, predictions, raw, contexts, _ = fixture()
    raw["entity-0"]["reason"][field] = value
    assert build_reasons(task, predictions, raw, contexts) == []


@pytest.mark.parametrize("change", ["point", "interval", "level", "label", "missing", "entity", "probs"])
def test_changed_final_answer_or_missing_raw_cannot_inherit_reason(change):
    task, predictions, raw, contexts, _ = fixture(classification=change == "label")
    if change == "point":
        predictions[0]["point_forecast"] = 3
    elif change == "interval":
        predictions[0]["interval"]["hi"] = 6
    elif change == "level":
        raw["entity-0"]["interval"]["level"] = 0.8
    elif change == "label":
        predictions[0]["label"] = "down"
    elif change == "missing":
        raw.clear()
    elif change == "entity":
        raw["entity-0"]["entity_id"] = "entity-1"
    else:
        raw["entity-0"]["label_probs"] = {"up": 0.5}
    assert build_reasons(task, predictions, raw, contexts) == []


def test_numeric_formatting_and_percent_level_are_not_changed_forecasts():
    task, predictions, raw, contexts, corpus = fixture()
    raw["entity-0"]["point_forecast"] = "2.500"
    raw["entity-0"]["interval"] = {"lo": "-1", "hi": "4", "level": 90}
    raw["entity-0"]["reason"]["answer_implication"] = "entity-0: point_forecast=2.5e0"
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == 1
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("text", ["entity-1: point_forecast=2.5", "entity-0: point_forecast=25",
                                 "entity-0: point_forecast=-2.5", "entity-0: point_forecast=NaN",
                                 "entity-0: point_forecast=2.5 but actually 9", "2.5", "up"])
def test_implication_must_unambiguously_match_final_answer(text):
    task, predictions, raw, contexts, _ = fixture()
    raw["entity-0"]["reason"]["answer_implication"] = text
    assert build_reasons(task, predictions, raw, contexts) == []


@pytest.mark.parametrize("change", ["task", "unknown", "inexact", "paraphrase", "future", "undated",
                                  "bad_date", "negative", "bool", "bad_end", "empty_context"])
def test_evidence_must_be_exact_dated_corpus_text(change):
    task, predictions, raw, contexts, _ = fixture()
    reason = raw["entity-0"]["reason"]
    chunk = contexts[0][0]
    if change in ("task", "unknown"):
        reason["doc_id"] = change
    elif change == "inexact":
        reason["quote"] = QUOTE.lower()
    elif change == "paraphrase":
        reason["premise"] = "Demand rose rapidly."
    elif change == "empty_context":
        contexts[0] = []
    else:
        fields = dict(vars(chunk))
        if change in ("future", "undated", "bad_date"):
            fields["doc_date"] = {"future": "2024-07-01", "undated": None, "bad_date": "2024-99-99"}[change]
        elif change == "negative":
            fields["span_start"] = -1
        elif change == "bool":
            fields["span_start"] = True
        else:
            fields["span_end"] += 1
        contexts[0] = [Chunk(**fields)]
    assert build_reasons(task, predictions, raw, contexts) == []


def test_invalid_candidate_does_not_discard_later_valid_reason():
    task, predictions, raw, contexts, corpus = fixture(2)
    raw["entity-0"]["reason"]["mechanism"] = "See u\u200bnits/ and TEAM NAME."
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == 1 and reasons[0]["reason_id"] == "r1"
    assert reasons[0]["scope"]["entities"] == ["entity-1"]
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("text", ["leaderboard", "canary", "see ://bad", "https://host/units/a",
                                 "\uff4ceaderboard", "team\u200b_id"])
def test_official_deny_list_filters_model_mechanisms(text):
    task, predictions, raw, contexts, _ = fixture()
    raw["entity-0"]["reason"]["mechanism"] = text
    assert build_reasons(task, predictions, raw, contexts) == []


def test_verbatim_premise_conservatively_rejects_deny_list_exemption():
    task, predictions, raw, contexts, corpus = fixture(quote="A canary in the coal mine was observed.")
    reasons = build_reasons(task, predictions, raw, contexts)
    assert reasons == []
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("chunk_is_fragment", [False, True])
def test_url_substring_cannot_gain_quote_exemption_by_rebasing(chunk_is_fragment):
    quote = "canary demand increased today."
    prefix = "https://example.org/"
    task, predictions, raw, contexts, _ = fixture(quote=quote, start=len(prefix))
    text = prefix + quote
    corpus = {"doc-0": rail.CorpusDoc("doc-0", text, "2024-06-01")}
    if not chunk_is_fragment:
        contexts[0] = [Chunk("doc-0", "2024-06-01", 0, len(text), text)]
    # Establish the authoritative verdict using the ORIGINAL document's URL.
    candidate = {"reason_id": "r1", **{k: raw["entity-0"]["reason"][k]
                 for k in ("premise", "mechanism", "answer_implication")},
                 "citations": [{"doc_id": "doc-0", "span_start": len(prefix), "span_end": len(text)}]}
    findings = rail.check_submitted_reasons(
        {"entity_predictions": predictions, "submitted_reasons": [candidate]}, corpus, task["cutoff_date"])
    assert "deny_list" in [f.code for f in findings]
    reasons = build_reasons(task, predictions, raw, contexts)
    assert reasons == []
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("quote", ["A \uff43anary demand observation.", "A ca\u200bnary demand observation.",
                                  "A https://example.org/units/data observation.",
                                  "A \uff1a//bad demand observation."])
def test_premise_filter_uses_official_folding_and_uri_rules(quote):
    task, predictions, raw, contexts, corpus = fixture(quote=quote)
    reasons = build_reasons(task, predictions, raw, contexts)
    assert reasons == []
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("length,expected", [(8000, 1), (8001, 0)])
def test_citation_character_cap(length, expected):
    quote = QUOTE + "x" * (length - len(QUOTE))
    task, predictions, raw, contexts, corpus = fixture(quote=quote)
    raw["entity-0"]["reason"]["premise"] = QUOTE
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == expected
    validate(task, predictions, reasons, corpus)


def test_reason_utf8_cap_and_continue_after_oversize():
    task, predictions, raw, contexts, corpus = fixture(3)
    raw["entity-0"]["reason"]["mechanism"] = "界" * 2200
    raw["entity-1"]["reason"]["mechanism"] = "é" * 2500
    raw["entity-2"]["reason"]["mechanism"] = "é" * 1000
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == 1 and reasons[0]["scope"]["entities"] == ["entity-1"]
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("tail", ["界" * 7900, "https://example.org/" + "x" * 7850])
def test_cumulative_evidence_bytes_include_unicode_and_url_masking(tail):
    task, predictions, raw, contexts, corpus = fixture(3, quote=QUOTE + tail)
    for row in raw.values():
        row["reason"]["premise"] = QUOTE
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == 1  # Two passages exceed 46,500 bytes after masking.
    validate(task, predictions, reasons, corpus)


def test_answer_cap_ignores_claims_but_counts_utf8_entity_ids():
    task, predictions, raw, contexts, _ = fixture()
    predictions[0]["claims"] = [{"claim": "x" * 10000}]
    assert build_reasons(task, predictions, raw, contexts)
    long_id = "界" * 1001
    task["entities"][0]["entity_id"] = long_id
    predictions[0]["entity_id"] = long_id
    raw[long_id] = raw.pop("entity-0")
    raw[long_id]["entity_id"] = long_id
    raw[long_id]["reason"]["answer_implication"] = f"{long_id}: point_forecast=2.5"
    assert build_reasons(task, predictions, raw, contexts) == []


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "bad_task", "bad_predictions",
                                  "bad_raw", "bad_contexts", "nan", "bool", "surrogate"])
def test_bad_top_level_inputs_fail_closed(change):
    task, predictions, raw, contexts, _ = fixture()
    if change == "missing":
        predictions.clear()
    elif change == "duplicate":
        predictions *= 2
    elif change == "extra":
        predictions.append({**predictions[0], "entity_id": "extra"})
    elif change == "bad_task":
        task = None
    elif change == "bad_predictions":
        predictions = [None]
    elif change == "bad_raw":
        raw = None
    elif change == "bad_contexts":
        contexts = []
    elif change == "surrogate":
        raw["entity-0"]["reason"]["mechanism"] = "\ud800"
    else:
        predictions[0]["point_forecast"] = float("nan") if change == "nan" else True
    assert build_reasons(task, predictions, raw, contexts) == []


def test_same_document_multiple_passages_keep_global_offsets():
    task, predictions, raw, contexts, corpus = fixture(2)
    text = "start " + QUOTE + " separator " + QUOTE
    for i, eid in enumerate(raw):
        start = text.find(QUOTE) if i == 0 else text.rfind(QUOTE)
        contexts[i] = [Chunk("shared", "2024-06-01", start, start + len(QUOTE), QUOTE)]
        raw[eid]["reason"]["doc_id"] = "shared"
    corpus["shared"] = rail.CorpusDoc("shared", text, "2024-06-01")
    reasons = build_reasons(task, predictions, raw, contexts)
    assert len(reasons) == 2
    assert [r["citations"][0]["span_start"] for r in reasons] == [text.find(QUOTE), text.rfind(QUOTE)]
    validate(task, predictions, reasons, corpus)


@pytest.mark.parametrize("extra,expected", [(0, 1), (1, 0)])
def test_exact_reason_byte_boundary(extra, expected):
    task, predictions, raw, contexts, corpus = fixture()
    reason = raw["entity-0"]["reason"]
    base = {"reason_id": "r1", "premise": QUOTE, "mechanism": "",
            "answer_implication": reason["answer_implication"]}
    reason["mechanism"] = "m" * (rail.MAX_REASON_BYTES - rail._compact_bytes([base]) + extra)
    result = build_reasons(task, predictions, raw, contexts)
    assert len(result) == expected
    validate(task, predictions, result, corpus)


@pytest.mark.parametrize("extra,expected", [(0, 1), (1, 0)])
def test_exact_answer_byte_boundary(extra, expected):
    task, predictions, raw, contexts, corpus = fixture()
    projection = {k: v for k, v in predictions[0].items() if k != "claims"}
    old_id = "entity-0"
    new_id = "e" * (len(old_id) + rail.MAX_ANSWER_BYTES - rail._compact_bytes([projection]) + extra)
    task["entities"][0]["entity_id"] = new_id
    predictions[0]["entity_id"] = new_id
    raw[new_id] = raw.pop(old_id)
    raw[new_id]["entity_id"] = new_id
    raw[new_id]["reason"]["answer_implication"] = f"{new_id}: point_forecast=2.5"
    result = build_reasons(task, predictions, raw, contexts)
    assert len(result) == expected
    validate(task, predictions, result, corpus)


@pytest.mark.parametrize("extra,expected", [(0, 1), (1, 0)])
def test_exact_evidence_byte_boundary_uses_original_not_rebased_offsets(extra, expected):
    task, predictions, raw, contexts, corpus = fixture(start=100000)
    projected = {"doc_id": "", "span_start": 100000, "span_end": 100000 + len(QUOTE),
                 "trusted_text": QUOTE, "reason_id": "r1"}
    doc_id = "d" * (rail.MAX_EVIDENCE_BYTES - rail._compact_bytes([projected]) + extra)
    raw["entity-0"]["reason"]["doc_id"] = doc_id
    contexts[0] = [Chunk(doc_id, "2024-06-01", 100000, 100000 + len(QUOTE), QUOTE)]
    corpus[doc_id] = rail.CorpusDoc(doc_id, " " * 100000 + QUOTE, "2024-06-01")
    result = build_reasons(task, predictions, raw, contexts)
    assert len(result) == expected
    validate(task, predictions, result, corpus)


def test_unicode_normalized_duplicate_reasons_are_omitted():
    task, predictions, raw, contexts, corpus = fixture(2)
    second_id = "ENTITY-0"
    task["entities"][1]["entity_id"] = second_id
    predictions[1]["entity_id"] = second_id
    raw[second_id] = raw.pop("entity-1")
    raw[second_id]["entity_id"] = second_id
    raw[second_id]["reason"]["answer_implication"] = f"{second_id}: point_forecast=2.5"
    raw[second_id]["reason"]["mechanism"] = MECHANISM.upper() + "\u200b"
    result = build_reasons(task, predictions, raw, contexts)
    assert len(result) == 1
    validate(task, predictions, result, corpus)


def test_substring_quote_uses_character_offsets_with_unicode_prefix():
    task, predictions, raw, contexts, corpus = fixture()
    prefix = "é界🙂 "
    text = prefix + QUOTE + " Extra text."
    contexts[0] = [Chunk("doc-0", "2024-06-01", 41, 41 + len(text), text)]
    corpus["doc-0"] = rail.CorpusDoc("doc-0", " " * 41 + text, "2024-06-01")
    result = build_reasons(task, predictions, raw, contexts)
    assert result[0]["citations"][0]["span_start"] == 41 + len(prefix)
    validate(task, predictions, result, corpus)


UNITS = sorted((UPSTREAM / "units").glob("*/task.json"))


@pytest.mark.parametrize("task_path", UNITS, ids=lambda p: p.parent.name)
def test_real_fixtures_pass_official_schema_and_checker(task_path):
    from retrieval import EvidenceIndex, build_index
    task = json.loads(task_path.read_text())
    corpus = rail.load_corpus(task_path.parent / "corpus")
    index = EvidenceIndex(build_index(task_path.parent / "corpus"), task["cutoff_date"])
    predictions, raw, contexts = [], {}, []
    for entity in task["entities"]:
        eid = entity["entity_id"]
        chunks = index.retrieve(task, entity, top_k=8)
        contexts.append(chunks)
        row = {"entity_id": eid, "interval": {"level": task["interval_level"], "lo": -1., "hi": 1.},
               "claims": []}
        if task["target"]["type"] == "classification":
            row["label"] = task["target"]["labels"][0]
            implication = f"{eid}: label={row['label']}"
        else:
            row["point_forecast"] = 0.0
            implication = f"{eid}: point_forecast=0"
        predictions.append(row)
        if chunks:
            chunk = chunks[0]
            quote = chunk.text[:180]
            row["claims"] = [{"doc_id": chunk.doc_id, "span_start": chunk.span_start,
                              "span_end": chunk.span_start + len(quote), "claim": quote}]
            raw[eid] = {**copy.deepcopy(row), "reason": {"premise": quote,
                        # Synthetic prose: validates plumbing, not forecast quality.
                        "mechanism": "The observation informs the forecast under uncertainty.",
                        "answer_implication": implication, "quote": quote, "doc_id": chunk.doc_id}}
    reasons = build_reasons(task, predictions, raw, contexts)
    assert 1 <= len(reasons) <= 3
    for reason in reasons:
        citation = reason["citations"][0]
        assert reason["premise"] == corpus[citation["doc_id"]].text[citation["span_start"]:citation["span_end"]]
    validate(task, predictions, reasons, corpus)


def test_public_fixtures_are_available():
    assert len(UNITS) == 11, "Use the pinned ede7381 public checkout for these tests"


def test_runtime_module_loads_without_site_packages_or_neural_libraries():
    script = ("import sys; sys.path[:0] = " + repr([str(ROOT), str(UPSTREAM)]) + "; "
              "import reasons; assert reasons.build_reasons({}, [], {}, []) == []; "
              "assert not any(x in sys.modules for x in ('torch', 'transformers', 'numpy'))")
    subprocess.run([sys.executable, "-S", "-B", "-c", script], check=True, timeout=20)
