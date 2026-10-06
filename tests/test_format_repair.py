import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from runtime import normalize_prediction

CLS = {"target": {"type": "classification", "labels": ["up", "down"]}, "interval_level": 0.9,
       "entities": [{"entity_id": "one"}]}
REG = {"target": {"name": "cpi_component_mom_first_print", "type": "regression"}, "interval_level": 0.9,
       "prompt": "For each component give a point_forecast (percent) of its month-over-month change.", "entities": [{"entity_id": "one"}]}


def row(**change):
    base = {"label": "up", "point_forecast": None, "interval": {"lo": 1, "hi": 3, "level": 0.9}, "evidence": []}
    base.update(change)
    return base


@pytest.mark.parametrize("label", ["UP", "Up", " up ", "up\n"])
def test_label_case_and_padding_map_to_the_one_allowed_label(label):
    assert normalize_prediction(row(label=label), CLS, {"entity_id": "one"}, [])["label"] == "up"


def test_a_label_matching_two_vocabulary_entries_is_refused():
    task = dict(CLS, target={"type": "classification", "labels": ["up", "Up"]})
    with pytest.raises(ValueError):
        normalize_prediction(row(label="UP"), task, {"entity_id": "one"}, [])
    with pytest.raises(ValueError):
        normalize_prediction(row(label="sideways"), CLS, {"entity_id": "one"}, [])


@pytest.mark.parametrize("level", ["90%", "90 %", "0.9"])
def test_the_task_level_written_as_text_is_accepted(level):
    result = normalize_prediction(row(interval={"lo": 1, "hi": 3, "level": level}), CLS, {"entity_id": "one"}, [])
    assert result["interval"] == {"level": 0.9, "lo": 1.0, "hi": 3.0}


@pytest.mark.parametrize("level", ["50%", "95 %"])
def test_another_level_written_as_text_is_still_refused(level):
    with pytest.raises(ValueError):
        normalize_prediction(row(interval={"lo": 1, "hi": 3, "level": level}), CLS, {"entity_id": "one"}, [])


def test_interval_as_a_two_number_list_and_swapped_bounds_are_repaired():
    assert normalize_prediction(row(interval=[1, 3]), CLS, {"entity_id": "one"}, [])["interval"]["hi"] == 3.0
    assert normalize_prediction(row(interval={"lo": 3, "hi": 1}), CLS, {"entity_id": "one"}, [])["interval"] == \
        {"level": 0.9, "lo": 1.0, "hi": 3.0}
    with pytest.raises(ValueError):
        normalize_prediction(row(interval=[1, 2, 3]), CLS, {"entity_id": "one"}, [])


def test_thousands_separators_parse_and_malformed_groups_do_not():
    reg = dict(REG, target={"name": "level", "type": "regression"}, prompt="Predict the level.")
    out = normalize_prediction(row(label=None, point_forecast="289,587.5",
                                   interval={"lo": "280,000.0", "hi": "300,000.0"}), reg, {"entity_id": "one"}, [])
    assert out["point_forecast"] == 289587.5 and out["interval"]["lo"] == 280000.0
    with pytest.raises(ValueError):
        normalize_prediction(row(label=None, point_forecast="1,23"), reg, {"entity_id": "one"}, [])


def test_a_percent_sign_is_dropped_only_when_the_output_unit_is_percent():
    out = normalize_prediction(row(label=None, point_forecast="0.3%", interval={"lo": "0.1%", "hi": "0.5%"}),
                               REG, {"entity_id": "one"}, [])
    assert out["point_forecast"] == 0.3 and out["interval"]["hi"] == 0.5
    ratio = dict(REG, target={"name": "bid_to_cover_ratio", "type": "regression"}, prompt="Predict the ratio.")
    with pytest.raises(ValueError):
        normalize_prediction(row(label=None, point_forecast="2.5%"), ratio, {"entity_id": "one"}, [])


def test_combined_formatting_preserves_explicit_ledger_and_raw_reply():
    from targets import target_contract
    entity = {'entity_id': 'one'}
    contract = target_contract(REG, entity)
    record = dict(target_name=contract['target_name'], forecast_period=contract['forecast_period'],
                  output_unit=contract['output_unit'], operation='identity',
                  point_input=1200, lo_input=1100, hi_input=1300)
    raw = row(point_forecast='1,200.0%', interval={'lo':'1,300.0%', 'hi':'1,100.0%', 'level':'90%'},
              target_record=record)
    before = copy.deepcopy(raw)
    result = normalize_prediction(raw, REG, entity, [])
    assert result['point_forecast'] == 1200
    assert result['interval'] == {'lo':1100, 'hi':1300, 'level':.9}
    assert raw == before
    record['point_input'] = 1100
    with pytest.raises(ValueError, match='arithmetic'):
        normalize_prediction(raw, REG, entity, [])


@pytest.mark.parametrize("value", ["1,23", "0,250", "-0,125", "00,123", "289,587", "1,250"])
@pytest.mark.parametrize("field", ["point_forecast", "lo", "hi"])
def test_ambiguous_comma_numbers_are_refused_in_forecasts_and_bounds(value, field):
    task = dict(REG, target={"name": "level", "type": "regression"}, prompt="Predict the level.")
    raw = row(label=None, point_forecast=2)
    if field == "point_forecast":
        raw[field] = value
    else:
        raw["interval"][field] = value
    with pytest.raises(ValueError):
        normalize_prediction(raw, task, {"entity_id": "one"}, [])


@pytest.mark.parametrize("value,expected", [("289,587.5", 289587.5), ("1,280,000", 1280000),
                                           ("1,300,000.0", 1300000), ("-1,280,000", -1280000)])
def test_unambiguous_grouped_numbers_keep_exact_scale(value, expected):
    task = dict(REG, target={"name": "level", "type": "regression"}, prompt="Predict the level.")
    result = normalize_prediction(row(label=None, point_forecast=value, interval=[value, value]),
                                  task, {"entity_id": "one"}, [])
    assert result["point_forecast"] == expected
    assert result["interval"]["lo"] == result["interval"]["hi"] == expected
