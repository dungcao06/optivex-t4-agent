"""Task-wide resolution must not replace an explicitly requested row release."""
import copy
import json
import os
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from targets import target_contract, validate_target
UNITS=Path(os.environ.get('T4_UPSTREAM_PATH', ROOT.parent/'track4-analysis-public-ede7381'))/'units'

def task():
    return {'prompt':'Predict next estimate in the release identified by resolving_release_date.',
            'target':{'name':'new_measure','type':'regression'},
            'cutoff_date':'2024-09-30','resolution_date':'2024-11-27'}

def test_explicit_row_release_is_forecast_period():
    t=task();e={'entity_id':'arbitrary','ref_month':'2024-08','resolving_release_date':'2024-10-04'}
    before=copy.deepcopy((t,e));c=target_contract(t,e)
    assert c['forecast_period']=='2024-10-04'
    assert c['task_resolution_date']=='2024-11-27'
    assert c['reference_period']=='2024-08'
    assert c['forecast_period_source']=='entity.resolving_release_date'
    assert (t,e)==before

@pytest.mark.parametrize('value',[None,'','2024-02-30','20241004','2024-10',True,
                                 '2024-09-30','2024-09-01','2024-12-01',['2024-10-04']])
def test_bad_or_inconsistent_row_dates_keep_legacy_horizon(value):
    assert target_contract(task(),{'resolving_release_date':value})['forecast_period']=='2024-11-27'

@pytest.mark.parametrize('prompt',['Forecast year-end value.',
                                  'Use previous_resolving_release_date as historical input.'])
def test_unreferenced_metadata_does_not_override_target_date(prompt):
    t=task();t['prompt']=prompt
    assert target_contract(t,{'resolving_release_date':'2024-10-04'})['forecast_period']=='2024-11-27'

def test_explicit_ledger_uses_requested_release():
    t=task();e={'resolving_release_date':'2024-10-04'};c=target_contract(t,e)
    prediction={'point_forecast':2,'interval':{'level':.9,'lo':1,'hi':3}}
    record=dict(target_name=c['target_name'],output_unit=c['output_unit'],forecast_period='2024-10-04',
                operation='identity',point_input=2,lo_input=1,hi_input=3)
    validate_target({'target_record':record},prediction,t,e)
    record['forecast_period']=t['resolution_date']
    with pytest.raises(ValueError,match='forecast_period'):
        validate_target({'target_record':record},prediction,t,e)
    validate_target({},prediction,t,e)

def test_public_macro_distinct_releases_for_same_reference_month():
    t=json.loads((UNITS/'t4-macrorev-20240930-us6/task.json').read_text())
    contracts={e['entity_id']:target_contract(t,e) for e in t['entities']}
    for e in t['entities']:
        c=contracts[e['entity_id']]
        assert c['forecast_period']==e['resolving_release_date']
        assert c['reference_period']==e['ref_month']
    assert contracts['PAYEMS_2024-08_20241004']['forecast_period']=='2024-10-04'
    assert contracts['PAYEMS_2024-08_20241101']['forecast_period']=='2024-11-01'

def test_other_public_families_keep_resolution_date():
    for path in UNITS.glob('*/task.json'):
        t=json.loads(path.read_text())
        if 'macrorev' in path.parent.name or 'auction-btc' in path.parent.name: continue
        for e in t['entities']:
            assert target_contract(t,e)['forecast_period']==t['resolution_date']

def test_prompt_separates_reference_release_and_global_resolution():
    import analyze
    t=task();e={'entity_id':'arbitrary','ref_month':'2024-08','resolving_release_date':'2024-10-04'}
    text=analyze.optivex_prompt(t,e,[])
    assert 'TASK-WIDE RESOLUTION DATE: 2024-11-27' in text
    assert 'REQUESTED ROW RELEASE DATE: 2024-10-04' in text
    assert 'REFERENCE PERIOD: 2024-08' in text
    assert 'do not substitute the task-wide resolution date' in text


def test_row_release_rewrites_target_record_instruction():
    import analyze
    text = analyze.optivex_prompt(task(), {'resolving_release_date': '2024-10-04'}, [])
    assert 'requested row release date; reference_period names' in text
    assert 'resolution date; the full task and entity fields specify' not in text


@pytest.mark.parametrize('key,value', [
    ('cutoff_date', None), ('cutoff_date', '2024-09-31'),
    ('resolution_date', '2024-11-31'), ('resolution_date', '20241127'),
])
def test_invalid_task_dates_do_not_bind_release(key, value):
    t = task()
    t[key] = value
    contract = target_contract(t, {'resolving_release_date': '2024-10-04'})
    assert 'forecast_period_source' not in contract
    assert contract['forecast_period'] == str(t.get('resolution_date', ''))


@pytest.mark.parametrize('reference', [None, '', '2024-13', '2024-2', '202402', 202408])
def test_invalid_reference_month_does_not_invent_observation_period(reference):
    contract = target_contract(task(), {'resolving_release_date': '2024-10-04',
                                        'ref_month': reference})
    assert contract['forecast_period'] == '2024-10-04'
    assert 'reference_period' not in contract


def test_release_on_task_resolution_is_valid():
    contract = target_contract(task(), {'resolving_release_date': '2024-11-27'})
    assert contract['forecast_period_source'] == 'entity.resolving_release_date'
    assert contract['forecast_period'] == '2024-11-27'
