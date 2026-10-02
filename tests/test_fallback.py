"""Fallback rows come from the unit's successful rows, never an unconditional placeholder."""
from __future__ import annotations

import json
import math
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


def test_huge_peer_values_stay_finite_and_serializable():
    row = fallback_prediction(REG, {"entity_id": "c"}, [peer(1e308, 1e308, 1e308)] * 2, CLAIM)
    assert all(math.isfinite(v) for v in (row["point_forecast"], row["interval"]["lo"], row["interval"]["hi"]))
    json.dumps(row, allow_nan=False)


def test_mixed_classification_peers_use_the_same_rows_for_point_and_interval():
    peers = [peer(50.0, 49.0, 51.0, "up"), {"label": "up", "interval": {"level": 0.9, "lo": 0.1, "hi": 0.5}}]
    row = fallback_prediction(CLS, {"entity_id": "c"}, peers, CLAIM)
    assert row["point_forecast"] == 50.0
    assert row["interval"] == {"level": 0.9, "lo": 49.0, "hi": 51.0}


def test_probability_fallback_without_success_is_in_domain():
    t={'prompt':'Give point_forecast as probability (0 to 1)', 'target':{'type':'classification','labels':['yes','no']}}
    r=fallback_prediction(t, {'entity_id':'lost'}, [], CLAIM)
    assert 0 <= r['interval']['lo'] <= r['point_forecast'] <= r['interval']['hi'] <= 1


def test_peer_values_with_different_units_are_not_mixed():
    t={'target':{'type':'regression'},'entities':[{'entity_id':'a','units':'dollars'},
        {'entity_id':'b','units':'thousands of persons'},{'entity_id':'c','units':'dollars'}]}
    p1={**peer(5,4,6),'entity_id':'a'};p2={**peer(1,.5,1.5),'entity_id':'b'}
    r=fallback_prediction(t,t['entities'][2],[p1,p2],CLAIM)
    assert (r['point_forecast'],r['interval']['lo'],r['interval']['hi'])==(5,4,6)
