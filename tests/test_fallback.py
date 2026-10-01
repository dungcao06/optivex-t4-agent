"""Fallback rows come from the unit's successful rows, never an unconditional placeholder."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fallback import fallback_prediction

CLAIM = [{"doc_id": "task", "span_start": 0, "span_end": 25, "claim": '{"entity_id": "c", "x": 1}'}]
REG = {"target": {"type": "regression"}, "interval_level": 0.9}
CLS = {"target": {"type": "classification", "labels": ["up", "down"]}, "interval_level": 0.9}


def peer(point, lo, hi, label=None):
    row = {"point_forecast": point, "interval": {"level": 0.9, "lo": lo, "hi": hi}}
    if label:
        row["label"] = label
    return row


def test_regression_uses_peer_medians_and_contains_the_point():
    row = fallback_prediction(REG, {"entity_id": "c"},
                              [peer(2.4, 2.1, 2.7), peer(2.6, 2.3, 2.9), peer(9.0, 8.0, 10.0)], CLAIM)
    assert row["point_forecast"] == 2.6
    assert row["interval"] == {"level": 0.9, "lo": 2.3, "hi": 2.9}
    assert row["claims"] == CLAIM and row["entity_id"] == "c"


def test_classification_uses_the_peer_majority_label():
    peers = [peer(0.7, 0.5, 0.9, "up"), peer(0.6, 0.4, 0.8, "up"), peer(0.2, 0.1, 0.4, "down")]
    row = fallback_prediction(CLS, {"entity_id": "c"}, peers, CLAIM)
    assert row["label"] == "up" and row["point_forecast"] == 0.6


def test_without_peers_the_legacy_values_remain():
    row = fallback_prediction(CLS, {"entity_id": "c"}, [], CLAIM)
    assert row["label"] == "up" and row["point_forecast"] == 0.0
    assert row["interval"] == {"level": 0.9, "lo": -1.0, "hi": 1.0}


def test_interval_always_contains_the_point():
    row = fallback_prediction(REG, {"entity_id": "c"}, [peer(5.0, 0.0, 1.0), peer(6.0, 0.0, 1.0)], CLAIM)
    assert row["interval"]["lo"] <= row["point_forecast"] <= row["interval"]["hi"]
