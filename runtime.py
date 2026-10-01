"""Budgeted inference and strict, exact-quote output construction.

The transport is intentionally dependency-free. Proxy environment variables are
left intact: the organizer supplies the only permitted model route.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from baselines.strong_rag_baseline.cli import _mock_reply
from claims import Ownership, build_claims, citable_note, load_ownership, task_row_claim
from fallback import fallback_prediction
from retrieval import EvidenceIndex, build_index

REQUEST_LIMIT = 25
BATCH_MARKER = "BATCH REQUESTS JSON:\n"


def _refuse_constant(value: str):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def parse_model_json(raw: str) -> dict:
    """Accept one final object, never braces inside reasoning or extra objects."""
    if not isinstance(raw, str):
        raise ValueError("Model content must be text")
    text = raw.strip()
    while text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ValueError("Unclosed reasoning block")
        text = text[end + len("</think>"):].strip()
    if "<think>" in text or "</think>" in text:
        raise ValueError("Unexpected reasoning delimiters")
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    value = json.loads(text, parse_constant=_refuse_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def finite_number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Expected a finite number")
    return number


def normalize_prediction(raw: dict, task: dict, entity: dict, chunks: list, *,
                         owners: Ownership | None = None) -> dict:
    """Validate the forecast fields; claims are only verbatim quotes the entity may cite."""
    if not isinstance(raw, dict):
        raise ValueError("Prediction must be an object")
    prediction = {"entity_id": entity["entity_id"]}
    if task["target"]["type"] == "classification":
        label = raw.get("label")
        if not isinstance(label, str) or label not in task["target"]["labels"]:
            raise ValueError("Label is not in the allowed vocabulary")
        prediction["label"] = label
    if task["target"]["type"] != "classification" or raw.get("point_forecast") is not None:
        prediction["point_forecast"] = finite_number(raw.get("point_forecast"))
    interval = raw.get("interval")
    if not isinstance(interval, dict):
        raise ValueError("Interval must be an object")
    lo, hi = finite_number(interval.get("lo")), finite_number(interval.get("hi"))
    if lo > hi:
        raise ValueError("Interval bounds are reversed")
    level = task.get("interval_level", 0.9)
    if interval.get("level") is not None:
        returned = finite_number(interval["level"])
        # A band at another level would be scored as the task's level; repair, never relabel.
        if not math.isclose(returned / 100 if returned > 1 else returned, level, abs_tol=1e-6):
            raise ValueError("Interval level differs from the task's declared level")
    prediction["interval"] = {"level": level, "lo": lo, "hi": hi}
    # Per-row calls cannot establish a cross-roster rank. The scorer uses the
    # comparable point_forecast vector; omit the optional, often invalid rank.
    claims = build_claims(raw.get("evidence"), chunks, task, entity["entity_id"], owners)
    if not claims:
        raise ValueError("No citable claim for this entity")
    prediction["claims"] = claims
    return prediction


class BudgetedClient:
    def __init__(self, system: str, deadline: float, mock: bool = False):
        self.system, self.deadline, self.mock = system, deadline, mock
        self.requests = 0
        endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
        if not mock and not endpoint.startswith(("http://", "https://")):
            raise ValueError("MODEL_ENDPOINT must be an HTTP(S) origin")
        self.url = endpoint + ("/chat/completions" if endpoint.endswith("/v1") else "/v1/chat/completions")
        self.timeout = min(45.0, max(0.1, float(os.environ.get("T4_MODEL_TIMEOUT_S", "35"))))
        self.seed = int(os.environ.get("QFBENCH_SEED", os.environ.get("T4_SEED", "42")))

    def complete(self, prompt: str) -> dict:
        remaining = self.deadline - time.monotonic()
        if self.requests >= REQUEST_LIMIT or remaining <= 0.1:
            raise ValueError("Unit inference budget exhausted")
        if len(prompt) > 110000:
            raise ValueError("Prompt exceeds the context safety bound")
        self.requests += 1
        if self.mock:
            if prompt.startswith(BATCH_MARKER):
                batch = json.loads(prompt[len(BATCH_MARKER):])
                return {"predictions": [dict(parse_model_json(_mock_reply(self.system, row["prompt"])),
                                             entity_id=row["entity_id"]) for row in batch]}
            return parse_model_json(_mock_reply(self.system, prompt))
        payload = {"model": os.environ.get("MODEL_NAME", "house"), "temperature": 0.0,
                   "seed": self.seed, "max_tokens": 4000,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "system", "content": self.system},
                                {"role": "user", "content": prompt}]}
        request = urllib.request.Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + os.environ.get("MODEL_TOKEN", "")},
                                         method="POST")
        with urllib.request.urlopen(request, timeout=min(self.timeout, remaining)) as response:
            body = response.read(2_000_001)
        if len(body) > 2_000_000:
            raise ValueError("Oversized model response")
        result = json.loads(body, parse_constant=_refuse_constant)
        try:
            if not isinstance(result, dict) or not isinstance(result.get("choices"), list) or not result["choices"]:
                raise ValueError("Invalid completion envelope")
            choice = result["choices"][0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise ValueError("Invalid completion envelope")
            if choice.get("finish_reason") == "length":
                raise ValueError("Model response truncated at token limit")
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Invalid completion envelope") from exc
        return parse_model_json(content)


def _safe_error(exc: Exception) -> str:
    # Never persist response bodies, request URLs, authorization, or tokens.
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code}"
    return type(exc).__name__


def run(task_path: Path, corpus_dir: Path, out_path: Path, system: str, prompt_builder, mock: bool = False) -> dict:
    started = time.monotonic()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = build_index(corpus_dir)
    index = EvidenceIndex(corpus, task["cutoff_date"])
    entities = task["entities"]
    contexts = [index.retrieve(task, entity, top_k=8) for entity in entities]
    owners = load_ownership(corpus_dir)
    prompts = [prompt_builder(task, entity, context) + citable_note(owners, entity["entity_id"], context)
               for entity, context in zip(entities, contexts)]
    client = BudgetedClient(system, started + 520, mock)
    predictions = [None] * len(entities)
    errors = {}
    batch_size = min(3, max(1, math.ceil(len(entities) / 20)))
    groups: list[list[int]] = []
    group: list[int] = []
    group_chars = 0
    for i, prompt in enumerate(prompts):
        # Account for JSON escaping and the repair suffix, not raw text alone.
        chars = len(json.dumps({"entity_id": entities[i]["entity_id"], "prompt": prompt}, ensure_ascii=False)) + 400
        if group and (len(group) >= batch_size or group_chars + chars > 109000):
            groups.append(group)
            group, group_chars = [], 0
        group.append(i)
        group_chars += chars
    if group:
        groups.append(group)
    # Primary pass before repairs prevents one troublesome row consuming the
    # unit budget. At most two repair rounds, still inside the shared 25 calls.
    for attempt in range(3):
        for group in groups:
            pending = [i for i in group if predictions[i] is None]
            if not pending or client.requests >= REQUEST_LIMIT or time.monotonic() >= client.deadline:
                continue
            rows = []
            for i in pending:
                prompt = prompts[i]
                if attempt:
                    prompt += "\nREPAIR: The previous response failed validation. Return complete valid JSON with finite numeric bounds and verbatim evidence quotes. Do not include reasoning."
                rows.append({"entity_id": entities[i]["entity_id"], "prompt": prompt})
            prompt = rows[0]["prompt"] if len(rows) == 1 else BATCH_MARKER + json.dumps(rows, ensure_ascii=False)
            try:
                raw = client.complete(prompt)
                if len(rows) == 1:
                    responses = {rows[0]["entity_id"]: raw}
                else:
                    values = raw.get("predictions")
                    if not isinstance(values, list):
                        raise ValueError("Batch response missing predictions")
                    responses = {}
                    for value in values:
                        if not isinstance(value, dict) or not isinstance(value.get("entity_id"), str):
                            raise ValueError("Malformed batch prediction")
                        if value["entity_id"] in responses:
                            raise ValueError("Duplicate entity in batch response")
                        responses[value["entity_id"]] = value
                for i in pending:
                    try:
                        predictions[i] = normalize_prediction(responses.get(entities[i]["entity_id"]), task, entities[i], contexts[i],
                                                               owners=owners)
                    except (ValueError, TypeError, OverflowError) as exc:
                        errors[str(entities[i]["entity_id"])] = _safe_error(exc)
            except (ValueError, TypeError, KeyError, OverflowError, OSError, http.client.HTTPException) as exc:
                for i in pending:
                    errors[str(entities[i]["entity_id"])] = _safe_error(exc)
    degraded_ids = []
    peers = [prediction for prediction in predictions if prediction is not None]
    for i, prediction in enumerate(predictions):
        if prediction is None:
            degraded_ids.append(entities[i]["entity_id"])
            claim = task_row_claim(task, entities[i]["entity_id"])
            predictions[i] = fallback_prediction(task, entities[i], peers, [claim] if claim else [])
    notes = {"agent": "optivex-evidence-council", "retrieval": "bounded-target-aware-passages",
             "model_requests": client.requests, "degraded_entities": len(degraded_ids), "mock": mock}
    if degraded_ids:
        notes.update(fallback_quality="unverified", degraded_entity_ids=degraded_ids,
                     quality_warning="Inference failed for these entities. Their forecasts are the median of this unit's successful rows and cite only the entity's task row; they are not evidence of predictive quality.",
                     errors={str(i): errors.get(str(i), "Budget exhausted") for i in degraded_ids})
    answer = {"task_id": task["task_id"], "schema_version": task.get("schema_version", "3"),
              "entity_predictions": predictions, "notes": notes,
              "evidence_trace": "Bounded pre-cutoff passages; exact quotation offsets. Exact matching establishes provenance, not semantic entailment. See notes for any inference failures."}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return answer


def main(system: str, prompt_builder, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="Canned replies for runtime testing only")
    args = parser.parse_args(argv)
    answer = run(args.task, args.corpus, args.out, system, prompt_builder, args.mock)
    print(f"wrote {args.out}: {len(answer['entity_predictions'])} entities; {answer['notes']['degraded_entities']} degraded; {answer['notes']['model_requests']} requests")
    return 0
