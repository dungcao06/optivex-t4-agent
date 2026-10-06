"""Independent auction timing acceptance tests; no observed outcomes or unit-ID dispatch."""
import copy
import json
import os
from pathlib import Path
import pytest
from targets import target_contract, validate_target


def scenario():
    entities = [dict(entity_id='first', auction_date='2030-11-04'),
                dict(entity_id='second', auction_date='2030-11-20')]
    return dict(prompt='Predict the bid-to-cover ratio for each auction from frozen evidence.',
                target=dict(name='bid_to_cover_ratio', type='regression'), cutoff_date='2030-10-31',
                resolution_date='2030-11-27', entities=entities)


def test_independent_synthetic_auction_uses_its_event_date():
    task = scenario()
    before = copy.deepcopy(task)
    for entity in task['entities']:
        c = target_contract(task, entity)
        assert c['forecast_period'] == entity['auction_date']
        assert c['forecast_period_source'] == 'entity.auction_date'
        assert c['task_resolution_date'] == task['resolution_date']
        assert 'reference_period' not in c
    assert task == before


def test_independent_public_auction_contract_and_ledger():
    upstream = Path(os.environ.get('T4_UPSTREAM_PATH', Path(__file__).resolve().parents[2] / 'track4-analysis-public-ede7381'))
    path = upstream / 'units/t4-auction-btc-202411-us7/task.json'
    task = json.loads(path.read_text())
    for entity in task['entities']:
        c = target_contract(task, entity)
        assert c['forecast_period'] == entity['auction_date']
        assert c['forecast_period_source'] == 'entity.auction_date'
        prediction = dict(point_forecast=2, interval=dict(level=.9, lo=1, hi=3))
        record = dict(target_name=c['target_name'], forecast_period=entity['auction_date'],
                      output_unit=c['output_unit'], operation='identity', point_input=2,
                      lo_input=1, hi_input=3)
        validate_target(dict(target_record=record), prediction, task, entity)


@pytest.mark.parametrize('bad_date', [None, '2030-11-31', '20301104', '2030-10-31',
                                     '2030-11-28', True])
def test_auction_invalid_dates_never_override(bad_date):
    task = scenario()
    entity = task['entities'][0]
    entity['auction_date'] = bad_date
    assert target_contract(task, entity)['forecast_period'] == task['resolution_date']


def test_auction_competing_date_does_not_guess_event():
    task = scenario()
    task['entities'][0]['settlement_date'] = '2030-11-08'
    assert target_contract(task, task['entities'][0])['forecast_period'] == task['resolution_date']


def test_auction_date_metadata_does_not_override_unrelated_prediction():
    task = scenario()
    task['prompt'] = 'Predict annual earnings growth using the supplied company histories.'
    assert target_contract(task, task['entities'][0])['forecast_period'] == task['resolution_date']


def test_reporting_date_does_not_replace_fiscal_forecast_period():
    task = scenario()
    task['prompt'] = 'Predict quarterly EPS for each company.'
    for entity in task['entities']:
        entity['expected_report_date'] = entity.pop('auction_date')
        assert target_contract(task, entity)['forecast_period'] == task['resolution_date']


def test_auction_initial_prompt_uses_event_not_release_language():
    from analyze import optivex_prompt
    task = scenario()
    prompt = optivex_prompt(task, task['entities'][0], [])
    assert 'REQUESTED AUCTION DATE: 2030-11-04' in prompt
    assert 'TASK-WIDE RESOLUTION DATE: 2030-11-27' in prompt
    assert 'requested auction date; the task-wide resolution date' in prompt
    assert 'resolution date; the full task and entity fields specify' not in prompt
    assert 'REQUESTED ROW RELEASE DATE' not in prompt
    assert 'may still affect pre-auction expectations' in prompt
    assert 'not as already observed outcomes' in prompt


def test_auction_review_preserves_distinct_event_dates():
    from review import build_review_prompt
    task = scenario()
    predictions = [dict(entity_id=e['entity_id'], point_forecast=2,
                        interval=dict(level=.9, lo=1, hi=3), claims=[])
                   for e in task['entities']]
    prompt = build_review_prompt(task, predictions, [[], []], None)
    assert 'requested_auction=2030-11-04' in prompt
    assert 'requested_auction=2030-11-20' in prompt
    assert 'TASK-WIDE RESOLUTION DATE: 2030-11-27' in prompt
    assert 'requested_release=' not in prompt
    assert 'may still affect pre-auction expectations' in prompt


def test_single_shared_auction_date_keeps_conservative_contract():
    task = scenario()
    task['entities'][1]['auction_date'] = task['entities'][0]['auction_date']
    assert 'forecast_period_source' not in target_contract(task, task['entities'][0])


def test_incidental_auction_background_does_not_change_eps_forecast():
    task = scenario()
    task['target']['name'] = 'annual_eps'
    task['prompt'] = ('Predict annual EPS. Auction bid-to-cover ratios are background '
                      'evidence about funding conditions, not the target.')
    assert target_contract(task, task['entities'][0])['forecast_period'] == task['resolution_date']


@pytest.mark.parametrize('name', ['demand', 'annual_eps', 'total_bids', 'yield'])
def test_ambiguous_or_different_target_names_keep_task_horizon(name):
    task = scenario()
    task['target']['name'] = name
    assert target_contract(task, task['entities'][0])['forecast_period'] == task['resolution_date']


@pytest.mark.parametrize('name', ['bid-to-cover ratio', 'Bid To Cover Ratio'])
def test_equivalent_explicit_ratio_names_bind_auction(name):
    task = scenario()
    task['target']['name'] = name
    assert target_contract(task, task['entities'][0])['forecast_period'] == '2030-11-04'


@pytest.mark.parametrize('prompt', [
    'Predict the average bid-to-cover ratio across all auctions for each tenor through resolution_date. auction_date is the first scheduled auction, not the forecast horizon.',
    'Predict the bid-to-cover ratio for each tenor using auction history.',
    'Predict the average bid-to-cover ratio across all auctions, starting with each auction listed.',
    'For each auction, predict the average bid-to-cover ratio through resolution_date.',
])
def test_aggregate_or_unbound_auction_horizon_is_not_overridden(prompt):
    from analyze import optivex_prompt
    from review import build_review_prompt
    task = scenario()
    task['prompt'] = prompt
    for entity in task['entities']:
        assert target_contract(task, entity)['forecast_period'] == task['resolution_date']
        text = optivex_prompt(task, entity, [])
        assert 'REQUESTED AUCTION DATE:' not in text
    predictions = [dict(entity_id=e['entity_id'], point_forecast=2,
                        interval=dict(level=.9, lo=1, hi=3), claims=[])
                   for e in task['entities']]
    text = build_review_prompt(task, predictions, [[], []], None)
    assert 'requested_auction=' not in text
    assert 'TASK-WIDE RESOLUTION DATE:' not in text
