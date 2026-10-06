"""Task-derived numerical contracts; no task-ID dispatch or outcome lookup.

Optional model records expose arithmetic for checking. A missing record remains compatible
with the published scaffold; an explicitly inconsistent record enters bounded repair.
"""
from __future__ import annotations
import math
import re
from datetime import date


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


def _iso_day(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _row_timing(task, entity):
    """Bind only an explicit release field named in the task, never infer from an ID."""
    if not re.search(r"(?<!\w)resolving_release_date(?!\w)", str(task.get('prompt', ''))):
        return {}
    requested = _iso_day(entity.get('resolving_release_date'))
    cutoff = _iso_day(task.get('cutoff_date'))
    resolution = _iso_day(task.get('resolution_date'))
    if requested is None or cutoff is None or resolution is None or not cutoff < requested <= resolution:
        return {}
    result = {'forecast_period': requested.isoformat(),
              'forecast_period_source': 'entity.resolving_release_date',
              'task_resolution_date': resolution.isoformat()}
    month = entity.get('ref_month')
    if isinstance(month, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}", month):
        if _iso_day(month + '-01') is not None:
            result['reference_period'] = month
    return result



def _auction_timing(task, entity):
    """Bind an explicitly dated auction target, never a company's reporting schedule."""
    prompt = str(task.get('prompt', '')).lower()
    target = task.get('target')
    if not isinstance(target, dict):
        return {}
    # Incidental auction evidence must not re-date a different prediction target.
    name = re.sub(r'[^a-z0-9]', '', str(target.get('name', '')).lower())
    if name != 'bidtocoverratio':
        return {}
    if not (re.search(r'\bauctions?\b', prompt) and
            re.search(r'\bbid[- ]to[- ]cover\b', prompt)):
        return {}
    # Bind only an explicit per-auction request, not an aggregate over a tenor/window.
    if not re.search(r'\b(?:for|of)\s+(?:each|every)\s+(?:(?!(?:tenor|issuer|using|based|through|until|after|before|across)\b)[a-z0-9.-]+\s+){0,8}auction\b', prompt):
        return {}
    if re.search(r'\b(?:average|aggregate|averaged|resolution_date)\b|\bacross\s+(?:all|multiple)\b', prompt):
        return {}
    requested = _iso_day(entity.get('auction_date'))
    cutoff = _iso_day(task.get('cutoff_date'))
    resolution = _iso_day(task.get('resolution_date'))
    if requested is None or cutoff is None or resolution is None or not cutoff < requested <= resolution:
        return {}
    # An additional event date can mean settlement or another horizon. Do not guess.
    for key, value in entity.items():
        if key != 'auction_date':
            other = _iso_day(value)
            if other is not None and cutoff < other <= resolution:
                return {}
    roster = task.get('entities')
    if not isinstance(roster, list):
        return {}
    dates = {_iso_day(row.get('auction_date')) for row in roster if isinstance(row, dict)}
    dates = {day for day in dates if day is not None and cutoff < day <= resolution}
    if requested not in dates or len(dates) < 2:
        return {}
    return {'forecast_period': requested.isoformat(),
            'forecast_period_source': 'entity.auction_date',
            'task_resolution_date': resolution.isoformat()}


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
            'cutoff_date': str(task.get('cutoff_date', '')), 'entity_fields': dict(entity),
            **_auction_timing(task, entity), **_row_timing(task, entity)}


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
