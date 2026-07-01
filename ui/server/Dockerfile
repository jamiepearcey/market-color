FROM node:20-bookworm-slim

ARG PORT=8787
ARG CELERITAS_BIN=/opt/celeritas/bin/celeritas

ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=${PORT} \
    CELERITAS_ROOT=/opt/celeritas \
    CELERITAS_BIN=${CELERITAS_BIN} \
    CELERITAS_API_TOKEN=change-me

WORKDIR /app

RUN mkdir -p /app/server /opt/celeritas /home/node/.celeritas-ui \
    && chown -R node:node /app /opt/celeritas /home/node/.celeritas-ui

COPY --chown=node:node server/*.mjs ./server/
COPY --chown=node:node server/.env.example ./server/.env.example
COPY --chown=node:node server/README.md ./server/README.md

USER node

VOLUME ["/opt/celeritas", "/home/node/.celeritas-ui"]

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD ["node", "-e", "const port=process.env.PORT||'8787';fetch(`http://127.0.0.1:${port}/health`).then((res)=>process.exit(res.ok?0:1)).catch(()=>process.exit(1));"]

CMD ["node", "server/celeritas-server.mjs"]
