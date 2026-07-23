# Install and run the backend server
FROM python:3.12-bookworm AS base

# Install additional locales
RUN apt-get update && apt-get install -y locales gettext  && rm -rf /var/lib/apt/lists/*
# Set locale to it_IT
RUN localedef -i it_IT -c -f UTF-8 -A /usr/share/locale/locale.alias it_IT.UTF-8
ENV LANG it_IT.utf8

FROM base AS python-env

WORKDIR /graphrag-aixlearning

ENV \
    # This prevents Python from writing out pyc files \
    PYTHONDONTWRITEBYTECODE=1 \
    # This keeps Python from buffering stdin/stdout \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN set -ex \
    && python3 -m venv $VIRTUAL_ENV \
    && $VIRTUAL_ENV/bin/pip install -U setuptools wheel pip uv

# Install Python packages from the hash-pinned lockfile for reproducible,
# supply-chain-verified production builds. requirements.txt is also copied
# because pyproject.toml resolves its dynamic `dependencies` from it during the
# editable install below (metadata generation reads the file even with --no-deps).
COPY requirements.lock.txt requirements.txt pyproject.toml README.md ./

RUN set -ex \
    && $VIRTUAL_ENV/bin/uv pip install -r requirements.lock.txt \
    && rm -rf /root/.cache/

FROM python-env AS api
ARG GIT_SHA
# Copy application files and install project in editable mode (deps already
# satisfied from the lockfile above, so --no-deps avoids re-resolving them).
COPY . .

# Fail the image build when a required, versioned retrieval asset was omitted
# from git or accidentally re-excluded by .dockerignore. These assets are
# immutable image content; only artifacts/media_cache is writable at runtime.
RUN set -ex \
    && test -s data/media/kg_neuro_media_pool.json \
    && test -s data/media/kg_udl_media_pool.json \
    && test -s artifacts/node2vec/neuro_node2vec_model.pkl \
    && test -s artifacts/node2vec/neuro_node2vec_embeddings.npz \
    && test -s artifacts/node2vec/udl_node2vec_model.pkl \
    && test -s artifacts/node2vec/udl_node2vec_embeddings.npz \
    && test -s artifacts/embeddings_cache/neuro_openai_embeddings.json \
    && test -s artifacts/embeddings_cache/udl_openai_embeddings.json

RUN $VIRTUAL_ENV/bin/pip install --no-deps -e .
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
ENV CODE_VERSION=${GIT_SHA}

# Defense-in-depth: run as a non-root user. Static retrieval assets stay
# read-only image content; the live-media cache is the only writable artifact
# path and may be mounted independently by Compose.
RUN useradd --create-home --uid 10001 aix \
    && mkdir -p /graphrag-aixlearning/artifacts/media_cache \
    && chown -R aix:aix /graphrag-aixlearning \
    && chmod -R a-w \
        /graphrag-aixlearning/data/media \
        /graphrag-aixlearning/artifacts/node2vec \
        /graphrag-aixlearning/artifacts/embeddings_cache
USER aix

# Serve on 8765 — the port Caddy proxies to (deploy/Caddyfile) and the compose
# healthcheck probes. 8765 is unprivileged, so the non-root user can bind it.
EXPOSE 8765

# Baked-in health check mirroring the compose-level one, so the image is
# self-describing even when run outside compose. curl ships in the bookworm base.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8765/api/v1/health || exit 1

# Shell form (not exec) so $API_CMD_ARGS expands — e.g. API_CMD_ARGS=--workers 2
# for the Phase B #39 multi-worker deploy.
CMD uvicorn aix.api.main:app --host 0.0.0.0 --port 8765 $API_CMD_ARGS
