# graphaixlearning — repo Makefile (Phase 2 reorg)
#
# Cross-platform note: requires GNU make.
#   - Linux / macOS / WSL: pre-installed
#   - Windows: `scoop install make` or `choco install make`, or run the
#     underlying commands directly (see `make help`).

.PHONY: help install api streamlit agent test test-int test-unit \
        ingest-neuro ingest-udl audit-neuro compile clean

PY  ?= python
PIP ?= pip

help:
	@echo ""
	@echo "graphaixlearning — Available make targets"
	@echo "=========================================="
	@echo ""
	@echo "  make install        - pip install requirements.txt"
	@echo ""
	@echo "  Runtime entry points:"
	@echo "    make api          - Run FastAPI on http://localhost:8000 (uvicorn --reload)"
	@echo "    make streamlit    - Run Streamlit UI at http://localhost:8501"
	@echo "    make agent        - Run interactive Agent CLI (apps/cli/run_agent.py)"
	@echo ""
	@echo "  Tests:"
	@echo "    make test         - Run all tests under tests/"
	@echo "    make test-int     - Run integration tests only (tests/integration/)"
	@echo "    make test-unit    - Run unit tests only (tests/unit/)"
	@echo ""
	@echo "  Data ingestion (requires .env Neo4j credentials):"
	@echo "    make ingest-neuro - Ingest data/kg/neuro/kg_neuro_neo4j.json"
	@echo "    make ingest-udl   - Ingest data/kg/udl/kg_udl_neo4j.json"
	@echo ""
	@echo "  Audit & maintenance:"
	@echo "    make audit-neuro  - Run KG audit for neuro domain"
	@echo "    make compile      - Syntax-check all Python (.py) files"
	@echo "    make clean        - Remove __pycache__ and .pyc files"
	@echo ""

install:
	$(PIP) install -r requirements.txt

api:
	uvicorn aix.api.main:app --reload --port 8000

streamlit:
	streamlit run apps/streamlit/main.py

agent:
	$(PY) apps/cli/run_agent.py

test:
	$(PY) -m pytest tests/

test-int:
	$(PY) -m pytest tests/integration/

test-unit:
	$(PY) -m pytest tests/unit/

ingest-neuro:
	$(PY) -m scripts.ingest.data_ingestion_neo4j --file data/kg/neuro/kg_neuro_neo4j.json --domain neuro

ingest-udl:
	$(PY) -m scripts.ingest.data_ingestion_neo4j --file data/kg/udl/kg_udl_neo4j.json --domain udl

audit-neuro:
	$(PY) -m scripts.audit.audit_domain_graph --domain neuro

compile:
	$(PY) -m compileall -q .

clean:
	@echo "Cleaning __pycache__ and .pyc files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."
