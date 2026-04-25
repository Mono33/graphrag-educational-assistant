"""Pytest configuration for graphaixlearning tests.

Ensures the project root is on ``sys.path`` so tests can import root-level
modules (``config``, ``graph_retriever``, ``llm_chain``, ``agent.*`` ...)
without needing the package to be pip-installed.

This shim is the testing counterpart to the ``sys.path`` shims at the top of
``apps/streamlit/main.py`` and ``apps/cli/run_agent.py``. Phase 3 will replace
all three with a proper editable install (``pip install -e .``).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
