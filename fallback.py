"""Fallback rows when inference fails, derived from the unit's successful rows.

Replaces the unconditional point 0 / interval [-1, 1] placeholder. The peer median is a
cross-entity estimate, not evidence about this entity; run notes still mark the row unverified.
"""
from __future__ import annotations

import statistics
from collections import Counter
from targets import target_contract

LEGACY_LABELS = ("inline", "neutral", "unchanged")


def fallback_prediction(task: dict, entity: dict, peers: list[dict], claims: list[dict]) -> dict:
    unit = target_contract(task, entity)['output_unit']
    entities = {e['entity_id']: e for e in task.get('entities', [])}
    compatible = [p for p in peers if p.get('entity_id') not in entities or
                  target_contract(task, entities[p['entity_id']])['output_unit'] == unit]
    scored = [p for p in compatible if "point_forecast" in p]
    if scored:
        # median_low returns a peer's own value: no averaging, so no overflow to infinity.
        point = float(statistics.median_low([p["point_forecast"] for p in scored]))
        lo = min(float(statistics.median_low([p["interval"]["lo"] for p in scored])), point)
        hi = max(float(statistics.median_low([p["interval"]["hi"] for p in scored])), point)
    elif unit == 'probability':
        point, lo, hi = 0.5, 0.0, 1.0
    else:
        point, lo, hi = 0.0, -1.0, 1.0
    row = {"entity_id": entity["entity_id"], "point_forecast": point,
           "interval": {"level": task.get("interval_level", 0.9), "lo": lo, "hi": hi},
           "claims": claims}
    target = task["target"]
    if target["type"] == "classification":
        labels = [p["label"] for p in compatible if p.get("label") in target["labels"]]
        row["label"] = (Counter(labels).most_common(1)[0][0] if labels else
                        next((x for x in LEGACY_LABELS if x in target["labels"]), target["labels"][0]))
    return row
