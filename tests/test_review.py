"""Joint whole-roster review: bounded prompt, sparse updates, atomic validation, no mutation."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze  # noqa: F401  Establish the pinned upstream import path.
from baselines.strong_rag_baseline.indexer import Chunk
from review import MAX_REVIEW_PROMPT_CHARS, MAX_UPDATES, apply_review, build_review_prompt

UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))
UNITS = sorted(p.parent for p in (UPSTREAM / "units").glob("t4-[a-z]*/task.json"))

CLAIM = {"doc_id": "task", "span_start": 0, "span_end": 20, "claim": '{"entity_id": "A", "x"'}
REG = {"task_id": "t", "cutoff_date": "2024-10-31", "prompt": "Rank markets by positioning change.",
       "target": {"name": "net_change_pct_oi", "type": "ranking"}, "interval_level": 0.9,
       "entities": [{"entity_id": "A", "name": "Corn"}, {"entity_id": "B", "name": "Wheat"},
                    {"entity_id": "C", "name": "Gold"}]}
ROWS = [{"entity_id": "A", "point_forecast": 1.0, "interval": {"level": 0.9, "lo": -2.0, "hi": 4.0}, "claims": [CLAIM]},
        {"entity_id": "B", "point_forecast": 3.0, "interval": {"level": 0.9, "lo": 0.0, "hi": 6.0}, "claims": [CLAIM]},
        {"entity_id": "C", "point_forecast": -1.0, "interval": {"level": 0.9, "lo": -5.0, "hi": 2.0}, "claims": [CLAIM]}]
CLS = dict(REG, target={"name": "eps_direction", "type": "classification", "labels": ["up", "down"]},
           prompt="Predict whether EPS rises year over year.")
CLS_ROWS = [dict(r, label=lab) for r, lab in zip(ROWS, ("up", "down", "up"))]
OWNERS = {"own_a": (frozenset({"A"}), False), "peer": (frozenset({"B"}), False), "shared": (frozenset(), True)}
TEXT = "Corn net long rose by 12,000 contracts in the week to October 22."
CONTEXTS = [[Chunk("own_a", "2024-10-25", 0, len(TEXT), TEXT), Chunk("peer", "2024-10-25", 0, len(TEXT), "PEER " + TEXT[5:])],
            [Chunk("shared", "2024-10-25", 0, len(TEXT), "SHARED " + TEXT[7:])], []]


def updates(*items) -> dict:
    return {"updates": list(items)}


def test_prompt_shows_the_whole_roster_and_only_citable_excerpts():
    prompt = build_review_prompt(REG, ROWS, CONTEXTS, OWNERS)
    for row in ROWS:
        assert row["entity_id"] in prompt
    assert "Corn" in prompt and "[-2, 4]" in prompt and TEXT in prompt
    assert "PEER" not in prompt and "SHARED" in prompt
    assert '"updates"' in prompt and str(MAX_UPDATES) in prompt
    assert len(prompt) <= MAX_REVIEW_PROMPT_CHARS


def test_prompt_is_bounded_and_refuses_oversized_or_mismatched_rosters():
    long = "word 12 " * 3000
    contexts = [[Chunk("own_a", "2024-10-25", 0, len(long), long)]] * 3
    assert len(build_review_prompt(REG, ROWS, contexts, OWNERS)) <= MAX_REVIEW_PROMPT_CHARS
    big = dict(REG, entities=[{"entity_id": f"E{i}"} for i in range(21)])
    big_rows = [dict(ROWS[0], entity_id=f"E{i}") for i in range(21)]
    assert build_review_prompt(big, big_rows, [[]] * 21, OWNERS) is None
    assert build_review_prompt(REG, ROWS[:2], CONTEXTS, OWNERS) is None
    assert build_review_prompt(REG, [ROWS[0], ROWS[0], ROWS[2]], CONTEXTS, OWNERS) is None


def test_valid_update_changes_only_the_point_and_preserves_everything_else():
    before = copy.deepcopy(ROWS)
    result = apply_review(updates({"entity_id": "C", "point_forecast": 1.5}), REG, ROWS)
    assert ROWS == before  # input never mutated
    assert [r["entity_id"] for r in result] == ["A", "B", "C"]
    assert result[2]["point_forecast"] == 1.5
    assert all(new["interval"] == old["interval"] and new["claims"] == old["claims"]
               for new, old in zip(result, ROWS))
    assert result[0] == ROWS[0] and result[1] == ROWS[1]


def test_no_updates_returns_an_identical_roster():
    assert apply_review(updates(), REG, ROWS) == ROWS


@pytest.mark.parametrize("raw", [
    updates({"entity_id": "C", "point_forecast": 2.5}),                          # outside original [-5, 2]
    updates({"entity_id": "Z", "point_forecast": 0.0}),                          # unknown entity
    updates({"entity_id": "A", "point_forecast": 0.0}, {"entity_id": "A", "point_forecast": 1.0}),
    updates(*[{"entity_id": e, "point_forecast": 0.0} for e in "ABC"] * 2),     # duplicates / too many
    updates({"entity_id": "A", "point_forecast": float("nan")}),
    updates({"entity_id": "A", "point_forecast": "high"}),
    updates({"entity_id": "A", "label": "up"}),                                   # label on a ranking unit
    updates({"entity_id": "A", "point_forecast": 0.0, "interval": {"lo": -9, "hi": 9}}),  # interval change
    updates({"entity_id": "A"}),                                                  # nothing to change
    {"updates": "x"}, {"predictions": []}, None, [], "text",
])
def test_any_invalid_update_rejects_the_whole_review(raw):
    assert apply_review(raw, REG, ROWS) is None


def test_more_than_max_updates_is_rejected():
    rows = [dict(ROWS[0], entity_id=f"E{i}") for i in range(MAX_UPDATES + 1)]
    task = dict(REG, entities=[{"entity_id": r["entity_id"]} for r in rows])
    raw = updates(*[{"entity_id": r["entity_id"], "point_forecast": 0.5} for r in rows])
    assert apply_review(raw, task, rows) is None


def test_classification_label_must_stay_in_vocabulary():
    result = apply_review(updates({"entity_id": "B", "label": "up"}), CLS, CLS_ROWS)
    assert result[1]["label"] == "up" and result[1]["interval"] == CLS_ROWS[1]["interval"]
    assert apply_review(updates({"entity_id": "B", "label": "sideways"}), CLS, CLS_ROWS) is None


def test_probability_targets_keep_the_point_inside_zero_one():
    task = dict(CLS, prompt="Give a label and a probability point_forecast for each firm.",
                target={"name": "credit_event_12m", "type": "classification", "labels": ["credit_event", "no_event"]})
    rows = [dict(r, label="no_event", point_forecast=0.2, interval={"level": 0.9, "lo": 0.0, "hi": 1.5}) for r in ROWS]
    assert apply_review(updates({"entity_id": "A", "point_forecast": 1.2}), task, rows) is None
    assert apply_review(updates({"entity_id": "A", "point_forecast": 0.4}), task, rows)[0]["point_forecast"] == 0.4


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_public_units_get_a_bounded_prompt_with_the_whole_roster(tmp_path, unit):
    from claims import load_ownership
    from retrieval import EvidenceIndex, build_index
    from runtime import run
    task = json.loads((unit / "task.json").read_text())
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "a.json",
                 analyze.SYSTEM_PROMPT, analyze.optivex_prompt, mock=True)
    index = EvidenceIndex(build_index(unit / "corpus"), task["cutoff_date"])
    contexts = [index.retrieve(task, e, top_k=8) for e in task["entities"]]
    prompt = build_review_prompt(task, answer["entity_predictions"], contexts, load_ownership(unit / "corpus"))
    assert prompt is not None and len(prompt) <= MAX_REVIEW_PROMPT_CHARS
    assert all(e["entity_id"] in prompt for e in task["entities"])
    assert apply_review(updates(), task, answer["entity_predictions"]) == answer["entity_predictions"]


def test_prompt_carries_resolution_date_full_target_and_output_units():
    task = dict(REG, resolution_date="2024-11-26",
                target={"name": "net_change_pct_oi", "type": "ranking", "unit": "pct_of_open_interest"})
    prompt = build_review_prompt(task, ROWS, CONTEXTS, OWNERS)
    assert "2024-11-26" in prompt
    assert json.dumps(task["target"], ensure_ascii=False) in prompt
    assert prompt.count("output_unit=") == len(ROWS)


def test_entity_metadata_is_kept_intact_not_sliced():
    long_field = "five-week window from the October 22 report to the November 26 report; " * 6
    task = dict(REG, entities=[dict(REG["entities"][0], window=long_field), *REG["entities"][1:]])
    prompt = build_review_prompt(task, ROWS, CONTEXTS, OWNERS)
    assert json.dumps(long_field, ensure_ascii=False) in prompt


def test_prompt_states_its_own_reply_format_over_any_system_format():
    prompt = build_review_prompt(REG, ROWS, CONTEXTS, OWNERS)
    assert "not the prediction format" in prompt


def test_cap_counts_every_character_and_refuses_an_oversized_head():
    huge = {"entity_id": "A", "name": "x" * (MAX_REVIEW_PROMPT_CHARS + 10)}
    task = dict(REG, entities=[huge, *REG["entities"][1:]])
    assert build_review_prompt(task, ROWS, CONTEXTS, OWNERS) is None
    long = "word 12 " * 3000
    contexts = [[Chunk("own_a", "2024-10-25", 0, len(long), long)]] * 3
    owners = {"own_a": (frozenset({"A", "B", "C"}), False)}
    assert len(build_review_prompt(REG, ROWS, contexts, owners)) <= MAX_REVIEW_PROMPT_CHARS
