"""Regression guard for immutable retrieval assets shipped in Docker."""

from pathlib import Path

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


def test_mandatory_docker_assets_exist_and_are_nonempty() -> None:
    """A deployable checkout must contain every versioned runtime asset."""
    missing = [p for p in MANDATORY_DOCKER_ASSETS if not (REPO_ROOT / p).is_file()]
    small = [
        p
        for p in MANDATORY_DOCKER_ASSETS
        if (REPO_ROOT / p).is_file() and (REPO_ROOT / p).stat().st_size <= 1024
    ]

    assert not missing, f"Mandatory Docker assets are missing: {missing}"
    assert not small, f"Mandatory Docker assets are unexpectedly small: {small}"
