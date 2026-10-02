"""Repair only impossible tails of a probability interval, preserving valid labels."""
import copy

import pytest

from test_targets import task, row
from targets import validate_target


PROB = task('Give point_forecast as probability (0 to 1)', type='classification', labels=['yes', 'no'])


@pytest.mark.parametrize('lo,hi,want', [(-1,1,(0,1)),(-.1,.8,(0,.8)),(.2,1.4,(.2,1)),(-2,3,(0,1))])
def test_probability_interval_is_intersected_with_its_known_support(lo, hi, want):
    prediction = row(.4, lo, hi)
    prediction['label'] = 'no'
    validate_target({}, prediction, PROB, {})
    assert (prediction['interval']['lo'], prediction['interval']['hi']) == want
    assert prediction['point_forecast'] == .4 and prediction['label'] == 'no'


@pytest.mark.parametrize('point,lo,hi', [(-.01,-1,1),(1.01,0,2),(.5,2,3),(.5,-3,-2),(.5,.8,.2),(.5,float('nan'),1)])
def test_invalid_points_and_disjoint_or_reversed_bands_still_require_repair(point, lo, hi):
    with pytest.raises(ValueError):
        validate_target({}, row(point,lo,hi), PROB, {})


def test_explicit_arithmetic_is_checked_before_interval_projection():
    record = {'target_name':'unseen_measure','forecast_period':'2028-06-30','output_unit':'probability',
              'operation':'identity','point_input':.4,'lo_input':-1,'hi_input':1}
    prediction = row(.4,-1,1)
    validate_target({'target_record':record}, prediction, PROB, {})
    assert prediction['interval']['lo'] == 0
    prediction = row(.4,-1,1)
    before = copy.deepcopy(prediction)
    with pytest.raises(ValueError, match='arithmetic'):
        validate_target({'target_record':dict(record, point_input=.9)}, prediction, PROB, {})
    assert prediction == before


def test_non_probability_intervals_and_skewed_forecast_quantiles_are_unchanged():
    prediction = row(.8,-1,2)
    before = copy.deepcopy(prediction)
    validate_target({}, prediction, task('Forecast a temperature change'), {})
    assert prediction == before
    prediction = row(.8,.1,.7)
    validate_target({}, prediction, PROB, {})
    assert prediction == row(.8,.1,.7)  # A mean need not lie between these quantiles.
