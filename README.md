## Executive summary (read this first)

This is Team `optivex`'s Agenthon 2026 Track 4 agent. It uses the official strong-RAG prompt and document types with a bounded participant runtime. Inference uses only the task table and frozen pre-cutoff corpus. Target-aware passages preserve original citation offsets; requests disable House-model thinking, obey a shared 25-call limit, and batch larger rosters. Malformed replies receive bounded repairs without aborting all entities.

Exact quotation matching establishes provenance, not semantic entailment. If inference cannot recover for an entity, its row takes the median forecast of the unit's successful rows, cites only the entity's task row and is marked `notes.fallback_quality="unverified"`. That is not a competitive prediction. A missing eligible corpus cannot be repaired by inventing evidence.

## Bounded roster review experiment

After successful primary forecasts, this candidate allows one non-thinking review request
for rosters of 2–20 entities when more than 60 seconds and one request remain. The review
compares all forecasts using the task fields and exact own/shared evidence excerpts. It can
change at most four entity forecasts: numeric points must stay inside their original
intervals and labels must stay within the declared vocabulary. Intervals and factual claims
are preserved. Any malformed or invalid update rejects the entire review; a timeout keeps
the primary answers without retry. Reasons for changed forecasts are removed. The same
25-request ceiling, 4,000-token output limit and request timeout apply.

For explicit probability targets, an otherwise valid interval is intersected with [0,1].
Nonfinite, reversed or wholly disjoint intervals and out-of-domain point forecasts still
fail. Explicit arithmetic is checked before repair. This avoids discarding valid labels
solely because an interval extends outside probability support; it does not impose a new
class prior or force a central interval to contain a skewed mean.

The hypothesis is that comparing related forecasts can improve ranking and consistency.
Mock and HTTP tests establish admission, evidence and budget behavior only. A Development
evaluation is required to measure forecasting performance against the observed 0.4067 base.

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

## Numeric identity retrieval

Retrieval preserves standalone unsigned hyphenated numeric terms, such as `2-Year`
and `10-Year`, without discarding their distinguishing number. Supported dash variants
and horizontal spacing around the dash share a token; complete period decimals keep
their fractional part. Ordinary number-word prose is not joined. Ambiguous numeric
tails fall back to existing tokens rather than being interpreted as a different
identity. Original corpus text, citation offsets and evidence limits are unchanged.

## Scorer 5.2.2 claim policy

Track 4 scorer 5.2.2 (public main `ede7381`) pays nothing for a true claim and multiplies a
unit's score by `1 - F / (F + min(T, 3E))` for false ones. A claim is false if it cites a
document the manifest does not label for its entity (nor marks shared), carries a figure no
cited span holds, exceeds 4,000 characters or 400 judge tokens, or is contradicted.

Claims are therefore short verbatim quotes (at most 400 characters; the densest 400-character
public window measured 280 judge tokens) from documents the read-only `corpus/manifest.json`
labels for the entity or marks shared. Duplicate spans collapse and at most three claims are
kept. If no model quote qualifies, the entity's own task-table row (`doc_id: "task"`) is
quoted. Peer documents stay in the prompt as context; the prompt names the citable ones.
The earlier table-context citation enrichment is retired.

A malformed or ungrounded citation now drops only that citation and keeps a valid forecast.
End-to-end tests run the scorer's own deterministic claim rules on every public unit. These
checks establish claim validity, not forecast quality, contradiction-judge outcomes or a
leaderboard gain.

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

The Dockerfile is standalone. During the build it fetches the official strong-RAG and citation guardrail
directories and MIT license at pinned commit `ede7381d8c1ba9d8c84068f9d142f5e093a33892`.
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

## Target checks and grounded reasons

The prompt declares task-derived output units, forecast period and permitted conversions.
Optional arithmetic records are checked against task baselines and denominators; probability
points and intervals must lie in [0, 1]. Retrieval can add one complementary absolute GAAP
EPS baseline while retaining the leading driver passages and existing evidence limits.
Fallbacks use compatible-unit peers, or [0, 1] for probabilities without peers. Other empty
peer cases retain the legacy placeholder and are marked unverified.

A successful first response may supply a grounded reason without an extra model request.
Only reasons that match the final prediction, original evidence and pinned deterministic
guardrails are submitted; otherwise the optional block is omitted. Premises containing
deny-list terms are conservatively omitted. This verifies structure and provenance, not
reasoning quality. Reasoning is evaluated separately on Final units, not Development.

## Numerical history context experiment

Starting from the best completed contract candidate (`2b8b222`, observed score 0.4025),
this candidate adds up to 1,200 characters of deterministic table summaries to each
entity prompt. Claude implemented the parser and initial tests; Codex integrated and
reviewed it. Only selected, manifest-authorized own/shared frozen documents are parsed.
Date-indexed rows yield the latest value, change since the previous available observation,
and median of up to six observations. Vintage grids keep the explicitly named reference
period fixed and summarize eligible vintage columns. Dates must be valid and no later
than both document date and task cutoff. Ambiguous chronology, nonfinite arithmetic,
unsupported tables and excessive inputs are omitted.

Records retain exact raw-cell offsets; the prompt shows their source envelope and column
name. The parser may read portions of a selected document outside retrieved excerpts.
These computed descriptions are explicitly non-citable context: claims and reasons still
must resolve to original retrieved quotations. Changes retain column units and are not
automatically converted to target units. Derived values display up to 12 significant digits;
raw latest values remain verbatim. Entity-matching columns are prioritized within the cap.

The runtime omits the whole optional block on error and records its error count. It adds no
model calls or forecast override, and keeps the existing retrieval, prediction validation,
intervals, reasons, call budget and output schema. Prompt changes can still help or hurt
forecasts. Local coverage is 46/78 public entities; arithmetic/provenance and HTTP tests do
not measure House-model accuracy. The 42-fold auction replay is a local cross-check against
an adjacent research artifact and is skipped in CI when that artifact is absent.
