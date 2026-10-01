"""Fallback rows when inference fails, derived from the unit's successful rows.

Replaces the unconditional point 0 / interval [-1, 1] placeholder. The peer median is a
cross-entity estimate, not evidence about this entity; run notes still mark the row unverified.
"""
from __future__ import annotations

import statistics
from collections import Counter

LEGACY_LABELS = ("inline", "neutral", "unchanged")


def fallback_prediction(task: dict, entity: dict, peers: list[dict], claims: list[dict]) -> dict:
    values = [p["point_forecast"] for p in peers if "point_forecast" in p]
    if values:
        point = float(statistics.median(values))
        lo = min(float(statistics.median([p["interval"]["lo"] for p in peers])), point)
        hi = max(float(statistics.median([p["interval"]["hi"] for p in peers])), point)
    else:
        point, lo, hi = 0.0, -1.0, 1.0
    row = {"entity_id": entity["entity_id"], "point_forecast": point,
           "interval": {"level": task.get("interval_level", 0.9), "lo": lo, "hi": hi},
           "claims": claims}
    target = task["target"]
    if target["type"] == "classification":
        labels = [p["label"] for p in peers if p.get("label") in target["labels"]]
        row["label"] = (Counter(labels).most_common(1)[0][0] if labels else
                        next((x for x in LEGACY_LABELS if x in target["labels"]), target["labels"][0]))
    return row
