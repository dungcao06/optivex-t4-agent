"""Response regressions that previously prevented answer.json from being written."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def parse(raw):
    return analyze.parse_model_json(raw)


def test_reasoning_braces_are_not_part_of_the_final_answer():
    raw = '<think>Consider {bad JSON} and {"label":"miss"}.</think>\n```json\n{"label":"beat"}\n```'
    assert parse(raw) == {"label": "beat"}


def test_unclosed_reasoning_is_not_mistaken_for_an_answer():
    with pytest.raises(ValueError):
        parse('<think>Candidate: {"label":"beat"}')


def test_trailing_analysis_cannot_select_an_ambiguous_object():
    with pytest.raises(ValueError):
        parse('{"label":"beat"}\n{"label":"miss"}')


def test_nested_object_with_quoted_braces():
    value = {"evidence": [{"quote": "a } brace { in text", "claim": "supported"}]}
    assert parse('```json\n' + json.dumps(value) + '\n```') == value


@pytest.mark.parametrize("raw", ['{"point_forecast":NaN}', '{"point_forecast":Infinity}', '[{"label":"beat"}]'])
def test_nonfinite_or_nonobject_payloads_are_refused(raw):
    with pytest.raises(ValueError):
        parse(raw)


def test_classification_can_omit_an_undefined_numeric_forecast():
    from runtime import normalize_prediction
    from baselines.strong_rag_baseline.indexer import Chunk
    raw = {"label": "beat", "point_forecast": None, "interval": {"lo": 1, "hi": 2},
           "evidence": [{"doc_id": "doc", "quote": "Evidence", "claim": "Context"}]}
    result = normalize_prediction(raw, {"target": {"type": "classification", "labels": ["beat"]},
                                   "entities": [{"entity_id": "one"}]},
                                  {"entity_id": "one"}, [Chunk("doc", "2024-01-01", 0, 8, "Evidence")])
    assert result["label"] == "beat"
    assert "point_forecast" not in result


TASK_90 = {"target": {"type": "regression"}, "interval_level": 0.9, "entities": [{"entity_id": "one"}]}


def _raw(level):
    interval = {"lo": 1, "hi": 3}
    if level is not None:
        interval["level"] = level
    return {"point_forecast": 2, "interval": interval, "evidence": []}


@pytest.mark.parametrize("level", [None, 0.9, 90])
def test_matching_or_absent_interval_level_is_accepted(level):
    from runtime import normalize_prediction
    result = normalize_prediction(_raw(level), TASK_90, {"entity_id": "one"}, [])
    assert result["interval"] == {"level": 0.9, "lo": 1.0, "hi": 3.0}


@pytest.mark.parametrize("level", [0.5, 50, 0.95, "0.5", True])
def test_a_different_interval_level_is_not_relabeled(level):
    """A 50% band relabeled as the task's 90% band would be scored as a too-narrow interval."""
    from runtime import normalize_prediction
    with pytest.raises(ValueError):
        normalize_prediction(_raw(level), TASK_90, {"entity_id": "one"}, [])
