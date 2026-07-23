"""Regression guard for immutable retrieval assets shipped in Docker."""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

MANDATORY_DOCKER_ASSETS = (
    "data/media/kg_neuro_media_pool.json",
    "data/media/kg_udl_media_pool.json",
    "artifacts/node2vec/neuro_node2vec_model.pkl",
    "artifacts/node2vec/neuro_node2vec_embeddings.npz",
    "artifacts/node2vec/udl_node2vec_model.pkl",
    "artifacts/node2vec/udl_node2vec_embeddings.npz",
    "artifacts/embeddings_cache/neuro_openai_embeddings.json",
    "artifacts/embeddings_cache/udl_openai_embeddings.json",
)


@pytest.mark.parametrize("relative_path", MANDATORY_DOCKER_ASSETS)
def test_mandatory_docker_asset_exists_and_is_nonempty(relative_path: str) -> None:
    """A deployable checkout must contain every versioned runtime asset."""
    asset = REPO_ROOT / relative_path

    assert asset.is_file(), f"Mandatory Docker asset is missing: {relative_path}"
    assert asset.stat().st_size > 1024, f"Mandatory Docker asset is unexpectedly small: {relative_path}"
