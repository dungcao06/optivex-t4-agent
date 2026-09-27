## Executive summary (read this first)

This directory is Team `optivex`'s first Agenthon 2026 Track 4 entry. It wraps the official strong-RAG starter components with an Optivex evidence-council prompt. The agent uses only the task table and frozen pre-cutoff corpus, grounds citations to exact spans, and emits the required classification, regression, or ranking output. The first milestone is a valid Development submission; numeric calibration and request-efficient batching come next.

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

The manual `Publish Agenthon image` GitHub Actions workflow builds the same context on Ubuntu,
checks the architecture and required interface label, runs `analyze --help`, executes the
official exemplar in mock mode, and only then pushes `ghcr.io/<owner>/optivex-t4-agent:dev`.
It records the immutable digest as a workflow artifact. The GHCR package must be set to public
and verified with an anonymous digest pull before submission.

## Secrets

Do not put the Agenthon Team Key, model token, registry credentials, or account credentials in this directory or image.
