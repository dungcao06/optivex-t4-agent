"""Only explicitly named binary-event probabilities imply the most probable label."""
import copy
import pytest
from targets import validate_target
from review import apply_review


def task(prompt=None, labels=None):
    return {'prompt': prompt or 'Give point_forecast = your predicted PROBABILITY of a credit event (0 to 1).',
            'target': {'name': 'future_event', 'type': 'classification',
                       'labels': labels or ['credit_event', 'no_event']},
            'entities': [{'entity_id': 'A'}, {'entity_id': 'B'}], 'interval_level': .9}


def row(p=.1, label='no_event', entity_id='A'):
    return {'entity_id': entity_id, 'label': label, 'point_forecast': p,
            'interval': {'level': .9, 'lo': 0., 'hi': 1.}, 'claims': []}


@pytest.mark.parametrize('p,original,expected', [(.9,'no_event','credit_event'),
    (.1,'credit_event','no_event'),(0,'credit_event','no_event'),(1,'no_event','credit_event'),
    (.5,'no_event','no_event'),(.5,'credit_event','credit_event')])
def test_primary_probability_selects_most_probable_explicit_event(p, original, expected):
    value = row(p, original)
    before = copy.deepcopy(value)
    validate_target({}, value, task(), {'entity_id':'A'})
    assert value['label'] == expected
    assert {k:v for k,v in value.items() if k != 'label'} == {k:v for k,v in before.items() if k != 'label'}


@pytest.mark.parametrize('prompt', [
    'Give point_forecast as probability (0 to 1).',
    'Give point_forecast = the mean EPS growth percent, with labels above or below zero.',
    'Give point_forecast = not the probability of a credit event.',
    'Give point_forecast = your predicted probability of a credit event or no event.',
    'Give point_forecast = your predicted probability of a credit event severity score.',
    'Give point_forecast = probability of a credit event. Also point_forecast = probability of no event.',
])
def test_ambiguous_mapping_or_numeric_mean_preserves_label(prompt):
    value = row(.9)
    validate_target({}, value, task(prompt), {'entity_id':'A'})
    assert value['label'] == 'no_event'


def test_explicit_complement_probability_and_reversed_label_order():
    value = row(.9, 'credit_event')
    validate_target({}, value, task('Give point_forecast = probability of no event (0 to 1).'), {})
    assert value['label'] == 'no_event'
    value = row(.9)
    validate_target({}, value, task(labels=['no_event','credit_event']), {})
    assert value['label'] == 'credit_event'


def test_nonbinary_or_missing_point_is_unchanged():
    value = row(.9)
    validate_target({}, value, task(labels=['credit_event','no_event','unknown']), {})
    assert value['label'] == 'no_event'
    value.pop('point_forecast')
    validate_target({}, value, task(), {})
    assert value['label'] == 'no_event'


def test_point_only_review_updates_scored_label_without_mutating_input():
    rows = [row(), row(entity_id='B')]
    before = copy.deepcopy(rows)
    got = apply_review({'updates':[{'entity_id':'A','point_forecast':.9}]}, task(), rows)
    assert got[0]['label'] == 'credit_event'
    assert got[0]['point_forecast'] == .9
    assert got[0]['interval'] == rows[0]['interval'] and got[0]['claims'] == rows[0]['claims']
    assert got[1] == rows[1] and rows == before


@pytest.mark.parametrize('update', [{'entity_id':'A','label':'credit_event'},
    {'entity_id':'A','label':'no_event','point_forecast':.9}])
def test_explicit_contradictory_review_is_rejected_atomically(update):
    rows = [row(), row(entity_id='B')]
    before = copy.deepcopy(rows)
    assert apply_review({'updates':[{'entity_id':'B','point_forecast':.2},update]},task(),rows) is None
    assert rows == before


def test_consistent_review_and_equal_probability_tie_are_accepted():
    rows = [row(), row(.5,entity_id='B')]
    got = apply_review({'updates':[{'entity_id':'A','label':'credit_event','point_forecast':.9},
                                  {'entity_id':'B','label':'credit_event'}]},task(),rows)
    assert got[0]['label'] == got[1]['label'] == 'credit_event'

@pytest.mark.parametrize('prompt', [
    'Do not give point_forecast = probability of a credit event. Report probability of no event instead.',
    'Do not give point_forecast = probability of a credit event (0 to 1).',
    'Give point_forecast = probability of a credit event, or no event, whichever is more likely.',
    'Give point_forecast = probability of a credit event (conditional on a recession).',
    'Give point_forecast = probability of a credit event (0 to 1), given a recession.',
    'If a recession occurs, give point_forecast = probability of a credit event (0 to 1).',
    'Give point_forecast = probability of a credit event (0 to 1), or no event if more likely.',
])
def test_negation_alternatives_and_conditions_do_not_map(prompt):
    spec = task(prompt)
    value = row(.9)
    validate_target({}, value, spec, {})
    assert value['label'] == 'no_event'
    got = apply_review({'updates':[{'entity_id':'A','point_forecast':.9}]}, spec,
                       [row(),row(entity_id='B')])
    assert got[0]['label'] == 'no_event'
