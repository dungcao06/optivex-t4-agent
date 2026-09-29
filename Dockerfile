FROM python:3.13-slim@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0 AS source

ARG TRACK4_COMMIT=2b307560c8183905a030dcb4cd26ce857a039cfd

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && git clone --filter=blob:none --no-checkout \
       https://github.com/Agenthon-2026/track4-analysis-public.git /tmp/track4 \
    && git -C /tmp/track4 checkout "$TRACK4_COMMIT" -- \
       baselines/strong_rag_baseline LICENSE


FROM python:3.13-slim@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0

LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.source="https://github.com/dungcao06/optivex-t4-agent"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY --from=source /tmp/track4/baselines/strong_rag_baseline /app/baselines/strong_rag_baseline
COPY --from=source /tmp/track4/LICENSE /usr/share/licenses/optivex/track4-analysis-public-LICENSE
COPY analyze.py /app/analyze.py
COPY runtime.py retrieval.py /app/
COPY LICENSE /usr/share/licenses/optivex/participant-LICENSE

ENTRYPOINT ["python", "/app/analyze.py"]
CMD ["analyze", "--help"]
