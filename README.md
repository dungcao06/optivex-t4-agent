## Executive summary (read this first)

This is Team `optivex`'s Agenthon 2026 Track 4 agent. It uses the official strong-RAG prompt and document types with a bounded participant runtime. Inference uses only the task table and frozen pre-cutoff corpus. Target-aware passages preserve original citation offsets; requests disable House-model thinking, obey a shared 25-call limit, and batch larger rosters. Malformed replies receive bounded repairs without aborting all entities.

Exact quotation matching establishes provenance, not semantic entailment. If inference cannot recover, the agent marks placeholder forecasts and context-only citations as `notes.fallback_quality="unverified"`. These are not a competitive prediction or a guarantee of passing the faithfulness gate. A missing eligible corpus cannot be repaired by inventing evidence.

## Quantity-and-period prompt experiment

This candidate adds an internal check for the requested quantity, units, denominator and
forecast period, distinct from an evidence item's observation period and publication date.
It asks the model to use compatible supplied quantities for changes, growth rates and ratios.
That prompt-only step left retrieval, model settings, runtime limits and the output schema unchanged.
This is a narrow Logistics/Time-inspired prompt experiment, not a validated LTF trading
strategy. Protocol tests can establish compatibility, not improved forecast quality; an
official comparison is required before claiming a competitive gain.

## Compact-table evidence experiment

The next candidate keeps recognized compact pipe-delimited tables together when they fit
the existing 2,200-character passage limit. Headers and dated values remain unchanged
original-text slices with exact citation offsets. An immediately preceding line is
included when it has at most 200 stripped characters, ends in `:` or contains the word
`unit`/`units`, and fits the same passage limit. This addresses lost table-tail evidence and supports the
quantity/period checks; it does not supply a numerical ledger or new model calls.

Recognition requires consistent nonempty columns, a textual header and data rows, with
an optional Markdown separator. Without a separator, each data row must contain a number.
Oversized, escaped-pipe or unrecognized tables retain ordinary bounded windows; flattened
filing tables are not reconstructed. Total evidence remains capped at 16,000 characters.
Ranking weights, issuer/cutoff filtering, prompts, model settings and runtime limits stay
unchanged, though different passage boundaries can change retrieval rankings. Generic
boundary tests and all six entities in the public rates fixture verify complete table
retrieval, not improved prediction quality or leaderboard score.

## Local run

From the official repository root (`track4-analysis-public/`):

```bash
.venv/bin/python ../optivex-t4-agent/analyze.py analyze \
  --mock \
  --task units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out /tmp/optivex-answer.json
```

Without `--mock`, the official harness supplies `MODEL_ENDPOINT`, `MODEL_NAME`, and `MODEL_TOKEN`.

## Container

The Dockerfile is standalone. During the build it fetches only the official strong-RAG
directory and MIT license at pinned commit `2b307560c8183905a030dcb4cd26ce857a039cfd`.
The Python base image and every write-capable GitHub Action are pinned by digest/commit.

Build locally from this directory when a Docker engine is available:

```bash
docker buildx build --platform linux/amd64 \
  -t YOUR_PUBLIC_REGISTRY/optivex-t4:dev .
```

The manual `Publish Agenthon image` GitHub Actions workflow runs the participant tests,
builds the same context on Ubuntu, checks architecture and interface, and exercises the
actual CLI over a local HTTP fixture for all 11 public units (78 entities), plus malformed
responses, request errors, citation failures and 30-entity rosters. It repeats the HTTP
suite inside the built container as an unprivileged user with a read-only filesystem,
dropped capabilities, a 64 MiB no-exec temporary filesystem, and process/resource ceilings.
Only then does it push `ghcr.io/<owner>/optivex-t4-agent:dev`.
It records the immutable digest as a workflow artifact. The GHCR package must be set to public
and verified with an anonymous digest pull before submission.

The HTTP fixture uses canned predictions, not the production House model. Tests verify
runtime/protocol/schema behavior, not predictive accuracy, semantic faithfulness, or rank.
The fixture container uses host networking to reach the local server; it does not reproduce
the organizer's proxy/firewall. Hosted-runner resource ceilings do not imply the runner has
the production system's physical CPU/RAM. A completed organizer run remains necessary.

## Tests

From this directory, with the sibling official environment installed:

```bash
../track4-analysis-public/.venv/bin/python -m pytest tests -q
```

Set `T4_UPSTREAM_PATH` if the pinned official checkout is elsewhere. Set `T4_TEST_IMAGE`
to run the HTTP integration suite inside a locally built Docker image on Linux.

## Secrets

Do not put the Agenthon Team Key, model token, registry credentials, or account credentials in this directory or image.
