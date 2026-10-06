"""Optional review exercises the real inference transport and output writer."""
import copy
import io
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import runtime


def exercise(tmp_path, monkeypatch, review_reply=None, *, count=2, fail_primary=False,
             remaining=520, starting_requests=0, review_error=None):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    text = 'Observed quantity is 10 widgets.'
    (corpus / 'series.json').write_text(json.dumps({
        'doc_id': 'series', 'doc_date': '2024-01-01', 'text': text}))
    entities = [{'entity_id': f'E{i}'} for i in range(count)]
    (corpus / 'manifest.json').write_text(json.dumps({'files': [
        {'path': 'corpus/series.json', 'entity_ids': [e['entity_id'] for e in entities]}]}))
    task = {'task_id': 'review-runtime', 'cutoff_date': '2024-01-02',
            'resolution_date': '2024-02-01', 'interval_level': .9,
            'prompt': 'Predict next month quantity in widgets.',
            'target': {'type': 'regression', 'name': 'quantity'}, 'entities': entities}
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(task))
    requests = []
    original_init = runtime.BudgetedClient.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.deadline = runtime.time.monotonic() + remaining
        self.requests = starting_requests
    monkeypatch.setattr(runtime.BudgetedClient, '__init__', init)
    def transport(request, **kwargs):
        requests.append(json.loads(request.data))
        if len(requests) <= count or fail_primary:
            reply = {} if fail_primary else {'point_forecast': 10,
                'interval': {'lo': 0, 'hi': 20},
                'evidence': [{'doc_id': 'series', 'quote': text}]}
        else:
            if review_error:
                raise review_error
            reply = review_reply
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(reply)}}]}).encode())
    monkeypatch.setenv('MODEL_ENDPOINT', 'http://fixture.invalid')
    monkeypatch.setattr(runtime.urllib.request, 'urlopen', transport)
    reason_inputs = []
    def reasons(task, predictions, raw, contexts):
        reason_inputs.append(copy.deepcopy(raw))
        return []
    monkeypatch.setattr(runtime, 'build_reasons', reasons)
    result = runtime.run(path, corpus, tmp_path / 'answer.json', analyze.SYSTEM_PROMPT, analyze.optivex_prompt)
    assert result == json.loads((tmp_path / 'answer.json').read_text())
    return result, requests, reason_inputs[0]


def response(a=12, b=10):
    return {'updates': [{'entity_id': 'E0', 'point_forecast': a},
                            {'entity_id': 'E1', 'point_forecast': b}]}


def test_review_updates_only_forecast_and_drops_changed_reason(tmp_path, monkeypatch):
    answer, requests, raw = exercise(tmp_path, monkeypatch, response())
    assert len(requests) == answer['notes']['model_requests'] == 3
    assert answer['notes']['roster_review'] == 'accepted'
    rows = answer['entity_predictions']
    assert [r['point_forecast'] for r in rows] == [12, 10]
    assert all(r['interval'] == {'level': .9, 'lo': 0, 'hi': 20} for r in rows)
    assert all(r['claims'] for r in rows)
    assert 'E0' not in raw and 'E1' in raw
    assert all(r['chat_template_kwargs']['enable_thinking'] is False for r in requests)
    assert all(r['max_tokens'] == 4000 for r in requests)
    assert requests[0]['messages'][0]['content'] == analyze.SYSTEM_PROMPT
    assert requests[-1]['messages'][0]['content'] == runtime.REVIEW_SYSTEM
    assert '"updates"' in requests[-1]['messages'][0]['content']


@pytest.mark.parametrize('reply', [{}, {'updates': 'invalid'}, response(30),
    {'updates': [{'entity_id': 'unknown', 'point_forecast': 12}]}])
def test_invalid_review_keeps_all_primary_rows(tmp_path, monkeypatch, reply):
    answer, requests, raw = exercise(tmp_path, monkeypatch, reply)
    assert len(requests) == 3
    assert answer['notes']['roster_review'] == 'rejected'
    assert [r['point_forecast'] for r in answer['entity_predictions']] == [10, 10]
    assert set(raw) == {'E0', 'E1'}


def test_review_transport_failure_has_no_retry_or_fallback(tmp_path, monkeypatch):
    answer, requests, raw = exercise(tmp_path, monkeypatch, review_error=TimeoutError())
    assert len(requests) == 3
    assert answer['notes']['roster_review'] == 'failed'
    assert answer['notes']['degraded_entities'] == 0
    assert [r['point_forecast'] for r in answer['entity_predictions']] == [10, 10]


@pytest.mark.parametrize('kwargs,expected_requests', [({'count': 1}, 1),
    ({'remaining': 59}, 2), ({'starting_requests': 23}, 2),
    ({'fail_primary': True}, 6)])
def test_review_skips_without_eligible_roster_or_budget(tmp_path, monkeypatch, kwargs, expected_requests):
    answer, requests, raw = exercise(tmp_path, monkeypatch, response(), **kwargs)
    assert len(requests) == expected_requests
    assert answer['notes']['roster_review'] == 'skipped'


def test_optional_prompt_bug_preserves_primary_answers(tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError('fixture-only prompt fault')
    monkeypatch.setattr(runtime, 'build_review_prompt', broken)
    answer, requests, raw = exercise(tmp_path, monkeypatch, response())
    assert len(requests) == 2
    assert answer['notes']['roster_review'] == 'failed'
    assert [r['point_forecast'] for r in answer['entity_predictions']] == [10, 10]


def test_late_review_is_discarded(tmp_path, monkeypatch):
    original_apply = runtime.apply_review
    clock = [0]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: clock[0])
    def late(*args):
        result = original_apply(*args)
        clock[0] = 521
        return result
    monkeypatch.setattr(runtime, 'apply_review', late)
    answer, requests, raw = exercise(tmp_path, monkeypatch, response())
    assert len(requests) == 3
    assert answer['notes']['roster_review'] == 'rejected'
    assert [r['point_forecast'] for r in answer['entity_predictions']] == [10, 10]
    assert set(raw) == {'E0', 'E1'}
