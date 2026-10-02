import copy
import math
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from targets import target_contract, validate_target, transform


def task(prompt='Predict EPS growth as a percent', **target):
    return {'prompt': prompt, 'target': {'name': 'unseen_measure', 'type': 'regression', **target},
            'resolution_date': '2028-06-30', 'cutoff_date': '2028-03-31'}


def row(point=25, lo=0, hi=50):
    return {'point_forecast': point, 'interval': {'level': .9, 'lo': lo, 'hi': hi}}


@pytest.mark.parametrize('op,value,base,denom,want', [
    ('percent_change', 2.5, 2, None, 25), ('bps_change', 4.2, 4, None, 20),
    ('change_pct_denominator', 150, 100, 1000, 5), ('change', 8, 10, None, -2)])
def test_explicit_transforms(op,value,base,denom,want):
    assert transform(value,op,base,denom) == pytest.approx(want)


@pytest.mark.parametrize('op,base,denom', [('percent_change',0,None),('change_pct_denominator',1,0),('unknown',1,1)])
def test_undefined_transform_is_rejected(op,base,denom):
    with pytest.raises(ValueError): transform(2,op,base,denom)


def test_contract_uses_prompt_not_entity_currency_for_growth():
    c=target_contract(task(), {'entity_id':'renamed','currency':'USD/share','prior_year_q_eps':2})
    assert c['output_unit']=='percent'
    assert c['entity_fields']['prior_year_q_eps']==2
    assert c['forecast_period']=='2028-06-30'


def test_probability_domain_is_explicit_not_every_classification():
    t=task('Report point_forecast = your predicted PROBABILITY (0 to 1)', type='classification')
    with pytest.raises(ValueError): validate_target({},row(-.2,-.4,1.5),t,{})
    validate_target({},row(20,10,30),task('Predict earnings in dollars',type='classification'),{})


def ledger(**changes):
    r={'target_name':'unseen_measure','forecast_period':'2028-06-30','output_unit':'percent',
       'operation':'percent_change','baseline_field':'prior_year_q_eps',
       'point_input':2.5,'lo_input':2,'hi_input':3}
    r.update(changes); return {'target_record':r}


def test_valid_ledger_matches_forecast_and_does_not_relabel_class():
    r=row();r['label']='down'
    validate_target(ledger(),r,task(),{'prior_year_q_eps':2})
    assert r['label']=='down'  # a skew distribution can have mean above baseline and majority below


@pytest.mark.parametrize('change', [dict(point_input=3),dict(forecast_period='2027-06-30'),
    dict(target_name='other'),dict(output_unit='USD/share'),dict(baseline_field='missing')])
def test_inconsistent_explicit_ledger_is_rejected(change):
    with pytest.raises(ValueError): validate_target(ledger(**change),row(),task(),{'prior_year_q_eps':2})


def test_changed_baseline_is_not_hidden_by_task_identity():
    with pytest.raises(ValueError): validate_target(ledger(),row(),task(),{'prior_year_q_eps':4})


def test_unknown_target_without_ledger_keeps_legacy_prediction():
    p=row(-12,-20,4); before=copy.deepcopy(p)
    validate_target({},p,task('Forecast a novel temperature anomaly in kelvin'),{})
    assert p==before


def test_nonfinite_intermediate_is_rejected():
    with pytest.raises(ValueError): transform(1e308,'bps_change',-1e308)


def test_runtime_routes_invalid_probability_to_repair():
    import analyze
    from runtime import normalize_prediction
    t=task('Give point_forecast as probability (0 to 1)',type='classification',labels=['yes','no'])
    entity={'entity_id':'arbitrary'};t['entities']=[entity]
    raw={'label':'yes','point_forecast':-.3,'interval':{'lo':-.4,'hi':.8},'evidence':[]}
    with pytest.raises(ValueError, match='Probability'):
        normalize_prediction(raw,t,entity,[])


def test_prompt_carries_explicit_contract_and_arithmetic_schema():
    import analyze
    t=task();e={'entity_id':'renamed','prior_year_q_eps':2}
    p=analyze.optivex_prompt(t,e,[])
    assert 'TARGET CONTRACT:' in p and 'target_record' in p
    assert 'percent_change' in p and 'baseline_field' in p


@pytest.mark.parametrize('operation,baseline', [('change','start_yield_pct'),('bps_change','maturity_years')])
def test_arithmetic_cannot_use_the_wrong_unit_or_baseline(operation,baseline):
    t=task('Predict the change in constant-maturity yield in basis points')
    e={'start_yield_pct':3.73,'maturity_years':10}
    c=target_contract(t,e)
    r={'target_name':c['target_name'],'forecast_period':c['forecast_period'],'output_unit':c['output_unit'],
       'operation':operation,'baseline_field':baseline,'point_input':3.5,'lo_input':3,'hi_input':4}
    vals=[transform(v,operation,e[baseline]) for v in (3.5,3,4)]
    with pytest.raises(ValueError): validate_target({'target_record':r},row(*vals),t,e)


def test_level_target_cannot_be_replaced_by_percent_growth():
    t=task('Give a point forecast of diluted EPS in USD/share')
    e={'prior_year_q_eps':2,'currency':'USD/share'}
    r=ledger(output_unit='USD/share')
    with pytest.raises(ValueError): validate_target(r,row(),t,e)
