"""One optional, non-thinking joint review of a unit's whole roster.

Primary forecasts come from independent per-entity calls, so nothing reconciles them across the
roster (a ranking, a yield curve, components against a headline, a shared base rate). This module
builds one bounded review prompt that shows every submitted answer side by side, and applies the
reply only if every proposed change is valid: at most a few sparse updates, each moving a point
inside that entity's ORIGINAL interval or a label within the vocabulary. Intervals, claims and all
other fields are preserved; anything invalid rejects the whole review and the baseline stands.
No model calls happen here; runtime.py decides whether budget allows the one request.
"""
from __future__ import annotations

import copy
import json
import math

from claims import Ownership, may_cite
from targets import target_contract

MAX_REVIEW_PROMPT_CHARS = 20_000
MAX_ROSTER = 20
MAX_UPDATES = 4
EXCERPT_CHARS = 400
_SKIP_FIELDS = ("corpus_ref",)


def _fmt(value: object) -> str:
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def _roster_matches(task: dict, predictions: list) -> bool:
    ids = [e.get("entity_id") for e in task.get("entities") or [] if isinstance(e, dict)]
    rows = [p.get("entity_id") if isinstance(p, dict) else None for p in predictions or []]
    return bool(ids) and len(ids) <= MAX_ROSTER and rows == ids and len(set(ids)) == len(ids)


def _answer(row: dict, classification: bool) -> str:
    parts = []
    if classification:
        parts.append(f"label={row.get('label')}")
    if "point_forecast" in row:
        parts.append(f"point_forecast={_fmt(row['point_forecast'])}")
    interval = row.get("interval", {})
    parts.append(f"interval=[{_fmt(interval.get('lo'))}, {_fmt(interval.get('hi'))}]")
    return ", ".join(parts)


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHARS:
        return text.strip()
    cut = text.rfind(" ", 0, EXCERPT_CHARS)
    return text[: cut if cut > 0 else EXCERPT_CHARS].strip()


ROW_RELEASE = "entity.resolving_release_date"
ROW_AUCTION = "entity.auction_date"


def _contract(task: dict, entity: dict) -> dict:
    try:
        return target_contract(task, entity)
    except Exception:  # noqa: BLE001 - a missing contract must not drop the review prompt.
        return {}


def _timing(contract: dict) -> str:
    """The row's own requested release, only when the target contract bound it explicitly."""
    if contract.get("forecast_period_source") == ROW_AUCTION:
        return f" | requested_auction={contract['forecast_period']}"
    if contract.get("forecast_period_source") != ROW_RELEASE:
        return ""
    reference = contract.get("reference_period")
    return f" | requested_release={contract['forecast_period']}" + (
        f" | reference_period={reference}" if reference else "")


def build_review_prompt(task: dict, predictions: list[dict], contexts: list[list],
                        owners: Ownership | None) -> str | None:
    """The whole-roster review request, or None when it cannot be built safely within the cap."""
    if not _roster_matches(task, predictions):
        return None
    target = task.get("target") or {}
    classification = target.get("type") == "classification"
    contracts = [_contract(task, entity) for entity in task["entities"]]
    timings = [_timing(contract) for contract in contracts]
    resolution = (f"TASK-WIDE RESOLUTION DATE: {task.get('resolution_date', '')} (each row's requested_release, "
                  "where listed, is the release to forecast)" if any(timings)
                  else f"RESOLUTION DATE: {task.get('resolution_date', '')}")
    if any(contract.get("forecast_period_source") == ROW_AUCTION for contract in contracts):
        resolution = (f"TASK-WIDE RESOLUTION DATE: {task.get('resolution_date', '')} "
                      "(each row's requested_auction or requested_release, where listed, "
                      "is the event to forecast; later scheduled events may still affect pre-auction expectations "
                      "but must not be treated as already observed outcomes)")
    head = [
        "ROSTER REVIEW REQUEST. All forecasts for this task are listed together below.",
        f"TASK: {task.get('prompt', '')}",
        "FULL TARGET SPECIFICATION: " + json.dumps(target, ensure_ascii=False),
        f"CUTOFF DATE: {task.get('cutoff_date', '')}; {resolution}; "
        f"INTERVAL LEVEL: {task.get('interval_level', 0.9)}",
        "SUBMITTED FORECASTS (entity metadata complete; output_unit from the target contract):",
    ]
    for entity, row, contract, timing in zip(task["entities"], predictions, contracts, timings):
        meta = json.dumps({k: v for k, v in entity.items() if k not in _SKIP_FIELDS}, ensure_ascii=False)
        unit = str(contract["output_unit"]) if "output_unit" in contract else "task-defined"
        head.append(f"- {entity['entity_id']}: {_answer(row, classification)} | "
                    f"output_unit={unit}{timing} | {meta}")
    tail = [
        "",
        'Reply format for THIS request: {"updates": [...]}, not the prediction format.',
        "Check the forecasts against each other: relative ordering and magnitudes, consistency",
        "across related entities, and a plausible share of each label across the roster. Change",
        f"at most {MAX_UPDATES} entities, only where the evidence clearly supports it. A new",
        "point_forecast must stay inside that entity's listed interval and use its output_unit"
        + ("; a new label must be one of the allowed labels" if classification else "")
        + ". Do not change intervals. Return ONE JSON object only:",
        '{"updates": [{"entity_id": "...", '
        + ('"label": "...", ' if classification else "")
        + '"point_forecast": number}]}',
        'Return {"updates": []} if no change is clearly better.',
    ]

    def render(excerpts: list[str]) -> str:
        block = ["CITABLE EXCERPTS (pre-cutoff, own or shared documents):", *excerpts] if excerpts else []
        return "\n".join(head + block + tail)

    if len(render([])) > MAX_REVIEW_PROMPT_CHARS:
        return None
    excerpts: list[str] = []
    seen = set()
    for entity, chunks in zip(task["entities"], contexts or []):
        for chunk in chunks or []:
            doc_id = getattr(chunk, "doc_id", None)
            if not isinstance(doc_id, str) or not may_cite(owners, doc_id, entity["entity_id"]):
                continue
            key = (doc_id, getattr(chunk, "span_start", None))
            if key in seen:
                continue
            line = f"[{entity['entity_id']} | {doc_id}] {_excerpt(chunk.text)}"
            if len(render(excerpts + [line])) <= MAX_REVIEW_PROMPT_CHARS:
                seen.add(key)
                excerpts.append(line)
            break  # One excerpt per entity keeps the prompt short.
    prompt = render(excerpts)
    return prompt if len(prompt) <= MAX_REVIEW_PROMPT_CHARS else None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def apply_review(raw: object, task: dict, predictions: list[dict]) -> list[dict] | None:
    """The full validated roster after sparse updates, or None to keep the baseline unchanged."""
    try:
        return _apply(raw, task, predictions)
    except Exception:  # noqa: BLE001 - optional review: never let it fail the unit.
        return None


def _apply(raw, task, predictions):
    if not _roster_matches(task, predictions) or not isinstance(raw, dict):
        return None
    items = raw.get("updates")
    if not isinstance(items, list) or len(items) > MAX_UPDATES:
        return None
    target = task["target"]
    classification = target["type"] == "classification"
    labels = target.get("labels") or []
    by_id = {e["entity_id"]: e for e in task["entities"]}
    result = copy.deepcopy(predictions)
    index = {row["entity_id"]: i for i, row in enumerate(result)}
    touched: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or set(item) - {"entity_id", "point_forecast", "label"}:
            return None
        entity_id = item.get("entity_id")
        if entity_id not in index or entity_id in touched or len(item) < 2:
            return None
        touched.add(entity_id)
        row = result[index[entity_id]]
        if "label" in item:
            if not classification or item["label"] not in labels:
                return None
            row["label"] = item["label"]
        if "point_forecast" in item:
            point = _finite(item["point_forecast"])
            lo, hi = _finite(row["interval"]["lo"]), _finite(row["interval"]["hi"])
            if point is None or lo is None or hi is None or not lo <= point <= hi:
                return None
            if (target_contract(task, by_id[entity_id])["output_unit"] == "probability"
                    and not 0 <= point <= 1):
                return None
            row["point_forecast"] = point
    return result
