"""Exercise the submitted CLI over real HTTP, optionally inside its Docker image.

T4_TEST_IMAGE selects the actual built image on Linux CI. Local tests otherwise
launch analyze.py in a fresh Python process. Canned forecasts verify runtime
contracts only; they cannot establish model quality or faithfulness scores.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from qfbench2_common.taskcard import schema_path

from http_fixture import ROOT, UNSUPPORTED_CLAIM, UPSTREAM, model_server
from baselines.strong_rag_baseline.indexer import build_index
from qfbench2_track_analysis.alignment import EntityRoster, align_predictions


UNITS = sorted(path.parent for path in (UPSTREAM / "units").glob("*/task.json"))
EXAMPLE = UPSTREAM / "units" / "t4-EXAMPLE-eps-beat"
SCHEMA = json.loads(schema_path("analysis.schema.json").read_text())


def run_agent(tmp_path: Path, unit: Path, mode: str = "valid", *, task: dict | None = None, endpoint_suffix: str = ""):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    output_dir.chmod(0o777)  # The container writes as uid 65534, never root.
    task_path = unit / "task.json"
    if task is not None:
        task_path = tmp_path / "task.json"
        task_path.write_text(json.dumps(task))
        task_path.chmod(0o644)
    output = output_dir / "answer.json"
    with model_server(mode) as (endpoint, requests):
        env = dict(os.environ, MODEL_ENDPOINT=endpoint + endpoint_suffix,
                   MODEL_NAME="fixture-house-model", MODEL_TOKEN="fixture-token-not-a-secret",
                   T4_MODEL_TIMEOUT_S="2", T4_MODEL_RETRIES="2", T4_UPSTREAM_PATH=str(UPSTREAM),
                   QFBENCH_SEED="314159", T4_SEED="999", PYTHONDONTWRITEBYTECODE="1")
        image = os.environ.get("T4_TEST_IMAGE")
        if image:
            command = ["docker", "run", "--rm", "--network", "host", "--read-only",
                       "--user", "65534:65534", "--cap-drop=ALL", "--security-opt", "no-new-privileges",
                       "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", "--pids-limit", "256",
                       "--ulimit", "nofile=1024:1024", "--ulimit", "nproc=256:256",
                       "--cpus", str(min(16, os.cpu_count() or 1)),
                       "--memory", "128g", "--memory-swap", "128g",
                       "-v", f"{task_path}:/input/task.json:ro",
                       "-v", f"{unit / 'corpus'}:/input/corpus:ro",
                       "-v", f"{output_dir}:/output"]
            for key in ("MODEL_ENDPOINT", "MODEL_NAME", "MODEL_TOKEN", "T4_MODEL_TIMEOUT_S", "T4_MODEL_RETRIES", "QFBENCH_SEED", "T4_SEED", "PYTHONDONTWRITEBYTECODE"):
                command.extend(["-e", key])
            command.extend([image, "analyze", "--task", "/input/task.json", "--corpus", "/input/corpus", "--out", "/output/answer.json"])
        else:
            command = [sys.executable, str(ROOT / "analyze.py"), "analyze", "--task", str(task_path),
                       "--corpus", str(unit / "corpus"), "--out", str(output)]
        result = subprocess.run(command, env=env, cwd=ROOT, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.is_file(), "A successful exit must have written answer.json"
        answer = json.loads(output.read_text(), parse_constant=lambda value: pytest.fail(f"Nonfinite JSON output: {value}"))
        captured = list(requests)
    return answer, captured


def assert_answer(answer: dict, task: dict, unit: Path):
    jsonschema.Draft202012Validator(SCHEMA).validate(answer)
    roster = EntityRoster.from_task(task)
    aligned = align_predictions(answer, roster, target_type=task["target"]["type"],
                                interval_level=task["interval_level"])
    assert aligned.count == roster.count
    assert answer["task_id"] == task["task_id"]
    rows = answer["entity_predictions"]
    assert [row["entity_id"] for row in rows] == [entity["entity_id"] for entity in task["entities"]]
    corpus = build_index(unit / "corpus")
    for row in rows:
        assert isinstance(row["point_forecast"], (int, float)) and math.isfinite(row["point_forecast"])
        lo, hi = row["interval"]["lo"], row["interval"]["hi"]
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
        if task["target"]["type"] == "classification":
            assert row["label"] in task["target"]["labels"]
        else:
            assert "label" not in row or isinstance(row["label"], str)
        for claim in row["claims"]:
            doc = corpus.doc_texts[claim["doc_id"]]
            assert 0 <= claim["span_start"] < claim["span_end"] <= len(doc)
            assert corpus.doc_dates[claim["doc_id"]] <= task["cutoff_date"]


def assert_request_contract(requests: list[dict]):
    assert 1 <= len(requests) <= 25
    for request in requests:
        assert request["path"] == "/v1/chat/completions"
        assert request["authorization"] == "Bearer fixture-token-not-a-secret"
        payload = request["payload"]
        assert payload["model"] == "fixture-house-model"
        assert payload["seed"] == 314159  # Organizer seed takes precedence over local override.
        assert payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
        assert 0 < payload.get("max_tokens", 0) <= 4000
        assert len(payload["messages"][-1]["content"]) <= 110000


@pytest.mark.parametrize("unit", UNITS, ids=lambda path: path.name)
def test_every_public_unit_over_http(tmp_path, unit):
    task = json.loads((unit / "task.json").read_text())
    answer, requests = run_agent(tmp_path, unit)
    assert_answer(answer, task, unit)
    assert_request_contract(requests)


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1/"])
def test_house_request_contract_for_origin_and_v1(tmp_path, suffix):
    answer, requests = run_agent(tmp_path, EXAMPLE, endpoint_suffix=suffix)
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert_request_contract(requests)


def test_reasoning_and_json_fences_do_not_crash(tmp_path):
    answer, requests = run_agent(tmp_path, EXAMPLE, "thinking")
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert answer.get("notes", {}).get("degraded_entities", 0) == 0
    assert_request_contract(requests)


@pytest.mark.parametrize("mode", ["malformed_first", "transient_http"])
def test_recoverable_response_gets_a_bounded_retry(tmp_path, mode):
    answer, requests = run_agent(tmp_path, EXAMPLE, mode)
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert 2 <= len(requests) <= 3
    assert answer.get("notes", {}).get("degraded_entities", 0) == 0
    assert_request_contract(requests)


def test_exhausted_http_errors_write_an_honestly_degraded_answer(tmp_path):
    answer, requests = run_agent(tmp_path, EXAMPLE, "unavailable")
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert answer["notes"]["degraded_entities"] == 1
    assert answer["notes"]["fallback_quality"] == "unverified"
    assert 1 <= len(requests) <= 3
    assert_request_contract(requests)


def test_ungrounded_quote_cannot_keep_its_claim(tmp_path):
    answer, requests = run_agent(tmp_path, EXAMPLE, "ungrounded")
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert UNSUPPORTED_CLAIM not in json.dumps(answer)
    assert answer["notes"]["degraded_entities"] == 1
    assert answer["notes"]["fallback_quality"] == "unverified"
    assert_request_contract(requests)


@pytest.mark.parametrize("mode", ["invalid_envelope", "truncated_envelope", "interval_list", "doc_id_list"])
def test_invalid_envelope_or_field_shapes_write_degraded_output(tmp_path, mode):
    answer, requests = run_agent(tmp_path, EXAMPLE, mode)
    assert_answer(answer, json.loads((EXAMPLE / "task.json").read_text()), EXAMPLE)
    assert answer["notes"]["degraded_entities"] == 1
    assert answer["notes"]["fallback_quality"] == "unverified"
    assert 1 <= len(requests) <= 3
    assert_request_contract(requests)


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_nonfinite_model_output_is_repaired(tmp_path, target_type):
    task = json.loads((EXAMPLE / "task.json").read_text())
    task["target"]["type"] = target_type
    if target_type != "classification":
        task["target"].pop("labels")
    answer, requests = run_agent(tmp_path, EXAMPLE, "nonfinite", task=task)
    assert_answer(answer, task, EXAMPLE)
    assert_request_contract(requests)


def test_per_row_ranks_are_not_trusted(tmp_path):
    task = json.loads((EXAMPLE / "task.json").read_text())
    task["target"] = {"name": "ranking_score", "type": "ranking"}
    task["entities"] = [dict(task["entities"][0], entity_id=f"entity-{i}", mktcap_bn=100 + i) for i in range(3)]
    answer, requests = run_agent(tmp_path, EXAMPLE, "invalid_rank", task=task)
    assert_answer(answer, task, EXAMPLE)
    ranks = [row.get("rank") for row in answer["entity_predictions"]]
    assert all(rank is None for rank in ranks) or sorted(ranks) == [1, 2, 3]
    assert_request_contract(requests)


@pytest.mark.parametrize("mode", ["valid", "malformed_first", "unavailable"])
def test_thirty_entities_fit_within_total_http_budget(tmp_path, mode):
    task = json.loads((EXAMPLE / "task.json").read_text())
    task["entities"] = [dict(task["entities"][0], entity_id=f"entity-{i}") for i in range(30)]
    answer, requests = run_agent(tmp_path, EXAMPLE, mode, task=task)
    assert_answer(answer, task, EXAMPLE)
    assert_request_contract(requests)
    if mode == "unavailable":
        assert answer["notes"]["degraded_entities"] == 30
    else:
        assert answer.get("notes", {}).get("degraded_entities", 0) == 0
