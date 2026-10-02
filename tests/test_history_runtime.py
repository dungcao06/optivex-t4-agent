"""Computed histories reach inference without overriding model forecasts."""
import io
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import runtime


def run_table(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = ("Observed quantity, units: widgets.\n"
            "date | quantity\n"
            "2024-01-01 | 1\n2024-01-02 | 2\n2024-01-03 | 4\n"
            "2024-01-04 | 8\n2024-01-05 | 19\n")
    (corpus / "series.json").write_text(json.dumps({
        "doc_id": "series", "doc_date": "2024-01-06", "text": text}))
    (corpus / "manifest.json").write_text(json.dumps({"files": [
        {"path": "corpus/series.json", "entity_ids": ["A"]}]}))
    task = {"task_id": "history-test", "cutoff_date": "2024-01-06",
            "resolution_date": "2024-02-01", "interval_level": 0.9,
            "target": {"type": "regression", "name": "quantity"},
            "entities": [{"entity_id": "A", "observed_count": 5}]}
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    requests = []

    def transport(request, **kwargs):
        requests.append(json.loads(request.data))
        # Transport double only: exercise the real budget and response handling.
        reply = {"point_forecast": 123, "interval": {"lo": 100, "hi": 150},
                 "evidence": [{"doc_id": "series", "quote": text}]}
        return io.BytesIO(json.dumps({"choices": [{"message": {
            "content": json.dumps(reply)}}]}).encode())

    monkeypatch.setenv("MODEL_ENDPOINT", "http://fixture.invalid")
    monkeypatch.setenv("MODEL_TOKEN", "fixture-token")
    monkeypatch.setattr(runtime.urllib.request, "urlopen", transport)
    answer = runtime.run(path, corpus, tmp_path / "answer.json",
                         analyze.SYSTEM_PROMPT, analyze.optivex_prompt)
    return answer, requests


def test_computed_change_reaches_model_without_overriding_forecast(tmp_path, monkeypatch):
    """The last change 19-8=11 is absent from the raw cells and must be computed."""
    answer, requests = run_table(tmp_path, monkeypatch)
    assert len(requests) == answer["notes"]["model_requests"] == 1
    prompt = requests[0]["messages"][-1]["content"]
    assert re.search(r"change[^\n]*?\b11(?:\.0)?\b", prompt, re.I)
    row = answer["entity_predictions"][0]
    assert row["point_forecast"] == 123
    assert row["interval"] == {"level": 0.9, "lo": 100, "hi": 150}
    assert all("median" not in claim["claim"].lower() for claim in row["claims"])


def test_optional_history_failure_preserves_valid_prediction(tmp_path, monkeypatch):
    """A parser failure must not turn an otherwise valid unit into a crash."""
    def fail(*args, **kwargs):
        raise RuntimeError("fixture parser failure")

    monkeypatch.setattr(runtime, "build_history", fail, raising=False)
    answer, requests = run_table(tmp_path, monkeypatch)
    assert answer["entity_predictions"][0]["point_forecast"] == 123
    assert answer["notes"]["degraded_entities"] == 0
    assert answer["notes"]["history_context_errors"] == 1
    assert len(requests) == 1


@pytest.mark.parametrize("block", ["x" * 1201, ["invalid history block"]])
def test_invalid_history_context_is_omitted_whole(tmp_path, monkeypatch, block):
    monkeypatch.setattr(runtime, "format_history", lambda *a, **kw: block)
    answer, requests = run_table(tmp_path, monkeypatch)
    assert answer["entity_predictions"][0]["point_forecast"] == 123
    assert answer["notes"]["history_context_errors"] == 1
    prompt = requests[0]["messages"][-1]["content"]
    assert "x" * 100 not in prompt
    assert "invalid history block" not in prompt
