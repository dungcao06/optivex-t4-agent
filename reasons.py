"""Optional first-pass reasons; no inference, file access or neural dependencies.

Integration contract:
* raw_by_entity maps entity_id to the original model prediction row, including
  optional ``reason: {premise, mechanism, answer_implication, doc_id, quote}``.
* contexts is a mapping of entity_id to retrieved Chunks, or a sequence of chunk
  lists in task entity order. These must be trusted, unmodified corpus excerpts.
* premise must be an exact substring of quote. mechanism is retained verbatim;
  this module never supplies missing causal reasoning.
  Premises with deny-list terms are omitted even when a full corpus quote might
  qualify for exemption: excerpt boundaries cannot establish that exemption.
* answer_implication must be ``ENTITY: point_forecast=NUMBER`` (regression or
  ranking) or ``ENTITY: label=LABEL`` (classification), without extra prose.
  Put the explanation in mechanism. Numeric formatting may differ from the row.
* predictions contains the final rows. Changed answers, missing raw rows and
  malformed reasons are omitted. Attach submitted_reasons only if nonempty.

Plan: validate roster/answers, resolve exact dated evidence, then admit at most
three candidates through the pinned official checker and original-offset caps.
The checker establishes structural validity, not the truth of model mechanisms.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date

from baselines.guardrails_example import citation_rail as rail

ANSWER_FIELDS = ("label", "point_forecast", "interval", "label_probs")
TEXT_FIELDS = ("premise", "mechanism", "answer_implication")


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite number")
    return number


def _answer(row: dict, task: dict) -> dict:
    """Compare semantic numeric values, including the runtime's level notation."""
    result = {key: row[key] for key in ANSWER_FIELDS if key in row}
    if task["target"]["type"] == "classification":
        if not isinstance(row.get("label"), str) or row["label"] not in task["target"]["labels"]:
            raise ValueError("Invalid label")
    elif "point_forecast" not in row:
        raise ValueError("Missing forecast")
    if "point_forecast" in result:
        result["point_forecast"] = _number(result["point_forecast"])
    band = row.get("interval")
    if not isinstance(band, dict):
        raise ValueError("Missing interval")
    level = _number(band.get("level", task.get("interval_level", 0.9)))
    if level > 1:
        level /= 100
    lo, hi = _number(band.get("lo")), _number(band.get("hi"))
    if lo > hi or level != _number(task.get("interval_level", 0.9)):
        raise ValueError("Invalid interval")
    result["interval"] = {"level": level, "lo": lo, "hi": hi}
    if "label_probs" in result:
        probs = result["label_probs"]
        if not isinstance(probs, dict):
            raise ValueError("Invalid probabilities")
        result["label_probs"] = {key: _number(value) for key, value in probs.items()}
        if any(not 0 <= value <= 1 for value in result["label_probs"].values()):
            raise ValueError("Invalid probability")
    return result


def _implication_matches(text: str, entity_id: str, answer: dict, task: dict) -> bool:
    field = "label" if task["target"]["type"] == "classification" else "point_forecast"
    prefix = f"{entity_id}: {field}="
    if not text.startswith(prefix):
        return False
    value = text[len(prefix):]
    return value == answer[field] if field == "label" else _number(value) == answer[field]


def _date(value: object) -> date:
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("Expected ISO date")
    return date.fromisoformat(value)


def _resolve(raw: dict, chunks: object, cutoff: date) -> tuple[dict, str, str] | None:
    doc_id, quote = raw.get("doc_id"), raw.get("quote")
    if (not isinstance(doc_id, str) or not doc_id or doc_id == "task"
            or not isinstance(quote, str) or not quote.strip()
            or len(quote) > rail.MAX_CITATION_CHARS or raw["premise"] not in quote):
        return None
    if not isinstance(chunks, (list, tuple)):
        return None
    for chunk in chunks:
        if getattr(chunk, "doc_id", None) != doc_id:
            continue
        try:
            text, start, end = chunk.text, chunk.span_start, chunk.span_end
            if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                    or start < 0 or end - start != len(text) or _date(chunk.doc_date) > cutoff):
                continue
            offset = text.find(quote)
            if offset >= 0:
                return ({"doc_id": doc_id, "span_start": start + offset,
                         "span_end": start + offset + len(quote)}, quote, chunk.doc_date)
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def _bytes(value: list) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          sort_keys=True, allow_nan=False).encode("utf-8")) - 2


def build_reasons(task: dict, predictions: list[dict], raw_by_entity: Mapping,
                  contexts: Mapping | Sequence) -> list[dict]:
    """Return zero to three validated reasons, in task order, without changing inputs.

    Empty means omit the optional field, never write ``submitted_reasons: []``.
    Raw/final answers must agree in every carried answer field, so repairs and
    fallbacks cannot silently inherit an explanation of a different forecast.
    Invalid candidates are skipped individually; invalid rosters omit all reasons.
    """
    try:
        return _build(task, predictions, raw_by_entity, contexts)
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError):
        # Optional explanations must not invalidate an otherwise usable answer.
        return []


def _build(task: dict, predictions: list[dict], raw_by_entity: Mapping,
           contexts: Mapping | Sequence) -> list[dict]:
    if not isinstance(task, dict) or not isinstance(predictions, list):
        return []
    if not isinstance(raw_by_entity, Mapping):
        return []
    entities = task.get("entities")
    if not isinstance(entities, list) or not all(isinstance(e, dict) for e in entities):
        return []
    ids = [e.get("entity_id") for e in entities]
    if not ids or not all(isinstance(e, str) and e for e in ids) or len(set(ids)) != len(ids):
        return []
    if not all(isinstance(p, dict) and isinstance(p.get("entity_id"), str) for p in predictions):
        return []
    by_id = {p["entity_id"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != set(ids):
        return []
    if not isinstance(contexts, Mapping):
        if not isinstance(contexts, (list, tuple)) or len(contexts) != len(ids):
            return []
        contexts = dict(zip(ids, contexts))
    cutoff = _date(task.get("cutoff_date"))
    # Count all answer fields, conservatively, as the official rail does. Claims
    # and notes are never part of this projection.
    projected = [{"entity_id": p["entity_id"], **{k: p[k] for k in ANSWER_FIELDS if k in p}}
                 for p in predictions]
    if _bytes(projected) > rail.MAX_ANSWER_BYTES:
        return []
    final_answers = {eid: _answer(by_id[eid], task) for eid in ids}
    accepted: list[dict] = []
    checked: list[dict] = []
    evidence: list[dict] = []
    corpus: dict[str, rail.CorpusDoc] = {}
    for eid in ids:
        if len(accepted) == rail.MAX_REASONS:
            break
        raw = raw_by_entity.get(eid)
        if not isinstance(raw, dict) or raw.get("entity_id", eid) != eid:
            continue
        reason = raw.get("reason")
        if not isinstance(reason, dict):
            continue
        if any(not isinstance(reason.get(k), str) or not reason[k].strip() for k in TEXT_FIELDS):
            continue
        try:
            # Never grant a quote exemption using rebased excerpts. A quote can
            # begin inside a URL in the original document, where masking removes
            # the apparent premise match. Check the premise as authored text,
            # including the official Unicode folding and malformed-URI rule.
            folded = rail.fold_for_detection(reason["premise"])[0].lower()
            masked = rail.fold_for_detection(rail.mask_reason_uris(reason["premise"]))[0]
            if any(term in folded for term in rail.DENY_LIST) or "://" in masked:
                continue
            if _answer(raw, task) != final_answers[eid]:
                continue
            if not _implication_matches(reason["answer_implication"], eid, final_answers[eid], task):
                continue
            resolved = _resolve(reason, contexts.get(eid), cutoff)
            if resolved is None:
                continue
            citation, quote, doc_date = resolved
            candidate = {"reason_id": f"r{len(accepted) + 1}",
                         **{k: reason[k] for k in TEXT_FIELDS},
                         "scope": {"entities": [eid]}, "citations": [citation]}
            # Rebase known exact passages for checker resolution, avoiding huge
            # padded allocations for a chunk far into a document. Output offsets
            # remain original; measure that evidence projection separately below.
            doc_id = citation["doc_id"]
            old = corpus.get(doc_id)
            if old is not None and old.doc_date != doc_date:
                continue
            prefix = old.text + "\n" if old else ""
            trial_corpus = {**corpus, doc_id: rail.CorpusDoc(doc_id, prefix + quote, doc_date)}
            local = {**candidate, "citations": [{"doc_id": doc_id,
                     "span_start": len(prefix), "span_end": len(prefix) + len(quote)}]}
            # _masked is the pinned rail's exact evidence-URI byte expansion.
            trial_evidence = evidence + [{**citation, "trusted_text": rail._masked(quote),
                                          "reason_id": candidate["reason_id"]}]
            if _bytes(trial_evidence) > rail.MAX_EVIDENCE_BYTES:
                continue
            answer = {"entity_predictions": predictions, "submitted_reasons": checked + [local]}
            if rail.check_submitted_reasons(answer, trial_corpus, task["cutoff_date"]):
                continue
        except (KeyError, TypeError, ValueError, OverflowError, UnicodeError):
            continue
        accepted.append(candidate)
        checked.append(local)
        evidence, corpus = trial_evidence, trial_corpus
    return accepted
