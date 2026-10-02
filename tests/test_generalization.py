"""Unseen Final units must not crash the agent or produce false claims.

Each public unit is perturbed the way a sealed family can differ (new entity and document
ids, unfamiliar target wording, no corpus manifest, a one-entity roster), its manifests and
digests are regenerated so the scorer accepts it, and the agent's mock answer must stay
schema-valid with no deterministic claim-rule finding. Mock replies test the contract, not
forecast quality.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze  # noqa: E402  Establish the pinned upstream import path.
from runtime import run  # noqa: E402

UPSTREAM = Path(os.environ.get("T4_UPSTREAM_PATH", ROOT.parent / "track4-analysis-public"))
UNITS = sorted(p.parent for p in (UPSTREAM / "units").glob("t4-[a-z]*/task.json")
               if (p.parent / "corpus" / "manifest.json").is_file())
VARIANTS = ("renamed_ids", "unfamiliar_target", "no_corpus_manifest", "single_entity")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rehash(unit: Path) -> None:
    """Recompute every root-manifest digest so the scorer accepts the perturbed unit."""
    manifest = json.loads((unit / "manifest.json").read_text())
    kept = []
    for entry in manifest["files"]:
        path = unit / entry["path"]
        if not path.is_file():
            continue
        data = path.read_bytes()
        entry["sha256"], entry["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
        kept.append(entry)
    manifest["files"] = kept
    _write_json(unit / "manifest.json", manifest)


def _relabel(entries: list[dict], ids: dict[str, str], docs: dict[str, str], roster: set[str]) -> None:
    for entry in entries:
        name = entry["path"].rsplit("/", 1)[-1][: -len(".json")]
        if name in docs:
            entry["path"] = entry["path"].rsplit("/", 1)[0] + "/" + docs[name] + ".json"
        if isinstance(entry.get("entity_ids"), list):
            entry["entity_ids"] = [ids.get(e, e) for e in entry["entity_ids"] if ids.get(e, e) in roster]


def perturb(source: Path, tmp_path: Path, variant: str) -> Path:
    unit = tmp_path / source.name
    shutil.copytree(source, unit)
    task = json.loads((unit / "task.json").read_text())
    corpus_manifest = json.loads((unit / "corpus" / "manifest.json").read_text())
    root_manifest = json.loads((unit / "manifest.json").read_text())
    ids = {e["entity_id"]: e["entity_id"] for e in task["entities"]}
    docs: dict[str, str] = {}
    if variant == "renamed_ids":
        ids = {old: f"ENT{i:02d}_X" for i, old in enumerate(ids)}
        stems = sorted(p.stem for p in (unit / "corpus").glob("*.json") if p.name != "manifest.json")
        docs = {old: f"DOCZ{j:02d}" for j, old in enumerate(stems)}
        for old, new in docs.items():
            path = unit / "corpus" / f"{old}.json"
            doc = json.loads(path.read_text())
            doc["doc_id"] = new
            path.unlink()
            _write_json(unit / "corpus" / f"{new}.json", doc)
        for entity in task["entities"]:
            entity["entity_id"] = ids[entity["entity_id"]]
    elif variant == "unfamiliar_target":
        task["target"]["name"] = "zeta_quantity_" + task["target"]["name"]
        task["prompt"] = ("Using only the frozen corpus, estimate the requested zeta quantity for each "
                          "row. Give the declared answer fields, a 90% interval, and cite passages.")
    elif variant == "single_entity":
        task["entities"] = task["entities"][:1]
        ids = {task["entities"][0]["entity_id"]: task["entities"][0]["entity_id"]}
    roster = {e["entity_id"] for e in task["entities"]}
    _relabel(corpus_manifest["files"], ids, docs, roster)
    _relabel([e for e in root_manifest["files"] if e["path"].startswith("corpus/")], ids, docs, roster)
    _write_json(unit / "task.json", task)
    _write_json(unit / "corpus" / "manifest.json", corpus_manifest)
    _write_json(unit / "manifest.json", root_manifest)
    if variant == "no_corpus_manifest":
        (unit / "corpus" / "manifest.json").unlink()
    _rehash(unit)
    return unit


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("source", UNITS, ids=lambda p: p.name)
def test_perturbed_unit_stays_valid_with_no_false_claim(tmp_path, source, variant):
    from baselines.guardrails_example.citation_rail import check_claim_rules
    from qfbench2_common.taskcard import schema_path
    from qfbench2_track_analysis.corpus import CorpusIndex

    unit = perturb(source, tmp_path, variant)
    CorpusIndex.from_unit(unit)  # The perturbation itself must be a well-formed unit.
    task = json.loads((unit / "task.json").read_text())
    answer = run(unit / "task.json", unit / "corpus", tmp_path / "answer.json",
                 analyze.SYSTEM_PROMPT, analyze.optivex_prompt, mock=True)

    schema = json.loads(schema_path("analysis.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(answer)
    assert [r["entity_id"] for r in answer["entity_predictions"]] == [e["entity_id"] for e in task["entities"]]
    findings = [f for f in check_claim_rules(answer, unit, token_counter=None)
                if f.code != "claim_tokens_unchecked"]
    assert findings == [], findings
    if variant == "no_corpus_manifest":
        assert {c["doc_id"] for r in answer["entity_predictions"] for c in r["claims"]} == {"task"}
