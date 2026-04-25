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

# Install Python packages (requirements.txt is read by pyproject.toml via dynamic deps)
COPY requirements.txt pyproject.toml README.md ./

RUN set -ex \
    && $VIRTUAL_ENV/bin/uv pip install -r requirements.txt \
    && rm -rf /root/.cache/

FROM python-env AS api
ARG GIT_SHA
# Copy application files and install project in editable mode
COPY . .
RUN $VIRTUAL_ENV/bin/pip install --no-deps -e .
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
ENV CODE_VERSION=${GIT_SHA}
CMD uvicorn aix.api.main:app --host 0.0.0.0 --port 80 $API_CMD_ARGS
