# syntax=docker/dockerfile:1
#
# market-color app image: Node API server (serves the built UI + /chat + /api)
# with the Python fact-RAG MCP server bundled in the same image. The API spawns
# the MCP over stdio using the system python3 (deps pre-installed here), so no
# `uv` fetch is needed at runtime. Postgres + Qdrant are separate services
# (see docker-compose.yml).

# ---------- UI build ----------
FROM node:20-bookworm-slim AS ui-build
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

# ---------- runtime: Node API + Python MCP ----------
FROM node:20-bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps: MCP server + the pipeline/bridge scripts (the full set).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      "mcp>=1.0" "qdrant-client>=1.15" "fastembed" "pyarrow" "duckdb>=1.0" \
      "python-dateutil" "httpx" "feedparser" "trafilatura" "googlenewsdecoder"

WORKDIR /app
# Node runtime deps (production only)
COPY ui/package.json ui/package-lock.json ./ui/
RUN cd ui && npm ci --omit=dev

# App code
COPY ui/server ./ui/server
COPY --from=ui-build /app/ui/dist ./ui/dist
COPY mcp ./mcp
COPY index_corpus.py index_facts.py graph_experiment.py bridge_facts_to_market_facts.py ./
COPY facts_work/facts.parquet ./facts_work/facts.parquet

ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=8787 \
    QDRANT_URL=http://qdrant:6333 \
    MARKET_FACTS_COLLECTION=market_facts \
    MARKET_MCP_COMMAND=python3 \
    MARKET_MCP_ARGS=/app/mcp/qdrant_facts_server.py

EXPOSE 8787
WORKDIR /app/ui
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-8787}/health" || exit 1
CMD ["node", "server/market-color-server.mjs"]
