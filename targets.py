"""Task-derived numerical contracts; no task-ID dispatch or outcome lookup.

Optional model records expose arithmetic for checking. A missing record remains compatible
with the published scaffold; an explicitly inconsistent record enters bounded repair.
"""
from __future__ import annotations
import math
import re


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError('Target arithmetic needs finite numbers')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError('Target arithmetic needs finite numbers') from exc
    if not math.isfinite(result):
        raise ValueError('Target arithmetic needs finite numbers')
    return result


def transform(value, operation, baseline=None, denominator=None):
    value = number(value)
    if operation == 'identity':
        return value
    base = number(baseline)
    if operation == 'change':
        result = value - base
    elif operation == 'bps_change':
        result = 100 * (value - base)
    elif operation == 'percent_change':
        if base == 0:
            raise ValueError('Percentage change needs a nonzero baseline')
        result = 100 * ((value - base) / base)
    elif operation == 'change_pct_denominator':
        denom = number(denominator)
        if denom <= 0:
            raise ValueError('Fixed denominator must be positive')
        result = 100 * ((value - base) / denom)
    else:
        raise ValueError('Unknown target transformation')
    return number(result)


def target_contract(task, entity):
    target = task.get('target', {})
    prompt = str(task.get('prompt', ''))
    lower = prompt.lower()
    # Only explicit output requests establish domains. Classification alone does not.
    probability = bool(re.search(r'(?:point[_ ]forecast|predicted|give|report).{0,60}probability', lower))
    if probability:
        unit = 'probability'
    elif 'basis points' in lower or 'yield change in bps' in lower:
        unit = 'basis_points'
    elif re.search(r'growth.{0,100}percent|point_forecast \(percent|return \(%\)|percent of.{0,80}open interest', lower):
        unit = 'percent'
    elif 'bid-to-cover ratio' in lower:
        unit = 'ratio'
    else:
        unit = str(target.get('unit', target.get('units', entity.get('unit', entity.get('units', entity.get('currency', 'task-defined'))))))
    conversions = [{'operation': 'identity'}]
    if unit == 'basis_points' and 'yield' in lower and 'start_yield_pct' in entity:
        conversions.append({'operation': 'bps_change', 'baseline_field': 'start_yield_pct'})
    if unit == 'percent' and 'growth' in lower and 'prior_year_q_eps' in entity and 'eps' in lower:
        conversions.append({'operation': 'percent_change', 'baseline_field': 'prior_year_q_eps'})
    # A declared general baseline also supports arbitrary percentage-growth quantities.
    if unit == 'percent' and 'growth' in lower:
        for key in ('baseline', 'baseline_value', 'prior_value'):
            if key in entity:
                conversions.append({'operation': 'percent_change', 'baseline_field': key})
    if unit == 'percent' and 'open interest' in lower and 'net position' in lower:
        for key in entity:
            if key.startswith('net_noncommercial_'):
                denom = 'open_interest_' + key.removeprefix('net_noncommercial_')
                if denom in entity:
                    conversions.append({'operation': 'change_pct_denominator', 'baseline_field': key,
                                        'denominator_field': denom})
    return {'allowed_conversions': conversions, 'target_name': target.get('name', ''), 'target_type': target.get('type', ''),
            'output_unit': unit, 'forecast_period': str(task.get('resolution_date', '')),
            'cutoff_date': str(task.get('cutoff_date', '')), 'entity_fields': dict(entity)}


def validate_target(raw, prediction, task, entity):
    contract = target_contract(task, entity)
    interval = prediction['interval']
    projected = None
    if contract['output_unit'] == 'probability':
        lo, hi = number(interval['lo']), number(interval['hi'])
        if 'point_forecast' in prediction and not 0 <= number(prediction['point_forecast']) <= 1:
            raise ValueError('Probability forecast must lie within [0,1]')
        if lo > hi or hi < 0 or lo > 1:
            raise ValueError('Probability interval must intersect [0,1] and be ordered')
        projected = (max(0.0, lo), min(1.0, hi))
    # Check explicit arithmetic against the original reply before support repair.
    _validate_record(raw, prediction, entity, contract)
    if projected is not None:
        interval['lo'], interval['hi'] = projected


def _validate_record(raw, prediction, entity, contract):
    interval = prediction['interval']
    record = raw.get('target_record')
    if record is None:
        return
    if not isinstance(record, dict):
        raise ValueError('target_record must be an object')
    for key in ('target_name', 'forecast_period', 'output_unit'):
        if record.get(key) != contract[key]:
            raise ValueError('Target record mismatches ' + key)
    operation = record.get('operation', 'identity')
    candidate = {'operation': operation}
    for key in ('baseline_field', 'denominator_field'):
        if record.get(key) is not None:
            candidate[key] = record[key]
    if candidate not in contract['allowed_conversions']:
        raise ValueError('Transformation or baseline is incompatible with target units')
    # Baselines and denominators are named task fields, not model-authored numbers.
    baseline = entity.get(record.get('baseline_field')) if isinstance(record.get('baseline_field'), str) else None
    denominator = entity.get(record.get('denominator_field')) if isinstance(record.get('denominator_field'), str) else None
    point = transform(record.get('point_input'), operation, baseline, denominator)
    ends = sorted(transform(record.get(k), operation, baseline, denominator) for k in ('lo_input', 'hi_input'))
    if 'point_forecast' not in prediction:
        raise ValueError('Numerical target record requires point_forecast')
    for want, actual in zip((point, *ends), (prediction['point_forecast'], interval['lo'], interval['hi'])):
        if not math.isclose(want, number(actual), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError('Forecast does not match explicit target arithmetic')
    # Never infer the class by thresholding a mean: a skewed distribution may disagree.
