#!/usr/bin/env python3
"""Measure the private material retrieval against the golden cases.

Why this exists
---------------
On 2026-09-16 the retrieval was tuned by looking at one lesson: the passages
seemed better afterwards, and nobody could say by how much or whether something
else had got worse. This script is the answer to "did that change help?" —
before/after numbers on the same cases, not an impression.

What it does, in order:

1. builds a throwaway index from ``data/eval/corpus/rag`` through the **real**
   parser, chunker and embedder — the same code the worker runs, so a defect in
   any of them shows up here rather than in a lesson;
2. runs every case in ``data/eval/rag/cases.yaml`` through the **real**
   retriever, with the ownership filter in place;
3. prints what was found, per case, with the numbers that matter.

Embeddings are cached on disk by ``(model, text)``, so the first run pays for the
corpus once and every later run is free and offline-repeatable. That is what
makes it reasonable to run this on every ranking change.

Metrics
-------
``recall@k``      of the pages a case expects, how many are in the top k.
``MRR``           1/rank of the first expected page — how high the right answer lands.
``boilerplate@5`` share of the top five that are pages a lesson must never quote
                  (a bibliography, a standards list). Zero is the only good value.
``documents``     whether every expected document contributed at least one passage:
                  a long document must not crowd out a short one.

Usage::

    python scripts/eval/run_private_rag.py                 # table + summary
    python scripts/eval/run_private_rag.py --offline       # cache only, no provider
    python scripts/eval/run_private_rag.py --check         # exit 1 below thresholds
    python scripts/eval/run_private_rag.py --case yalta -v # one case, with its passages
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CASES_FILE = REPO_ROOT / "data" / "eval" / "rag" / "cases.yaml"
CACHE_DIR = REPO_ROOT / "data" / "eval" / "rag" / ".cache"

#: Thresholds ``--check`` enforces. Deliberately modest: they are a floor that
#: stops a regression reaching a teacher, not a target to optimise against.
MIN_MEAN_RECALL = 0.80
MAX_BOILERPLATE = 0.0

OWNER = uuid.UUID("11111111-1111-1111-1111-111111111111")


# ---------------------------------------------------------------------------
# Embeddings, cached
# ---------------------------------------------------------------------------


class CachedEmbeddings:
    """The provider, with a disk cache — injected where the code already allows it.

    ``embed_texts`` and ``PrivateMaterialRetriever.search`` both accept a client,
    so nothing in the library needs to know this exists.
    """

    def __init__(
        self, model: str, *, offline: bool, dimensions: int | None = None, one_per_request: bool = False
    ) -> None:
        self.model = model
        self.offline = offline
        #: Asked of models that can shorten their vectors (Matryoshka), so a
        #: candidate is measured at the size the index actually stores.
        self.dimensions = dimensions
        self.one_per_request = one_per_request
        suffix = f"@{dimensions}" if dimensions else ""
        self.directory = CACHE_DIR / (model.replace("/", "_") + suffix)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self._client = None

    @property
    def embeddings(self) -> CachedEmbeddings:  # matches the provider's shape
        return self

    def _path(self, text: str) -> Path:
        return self.directory / f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.json"

    async def create(self, *, model: str, input: list[str]):  # noqa: A002 — provider's name
        vectors: list[list[float]] = []
        missing: list[tuple[int, str]] = []

        for index, text in enumerate(input):
            path = self._path(text)
            if path.exists():
                vectors.append(json.loads(path.read_text(encoding="utf-8")))
                self.hits += 1
            else:
                vectors.append([])
                missing.append((index, text))

        if missing:
            if self.offline:
                raise SystemExit(
                    f"{len(missing)} text(s) are not in the cache and --offline was given. "
                    "Run once without it to populate the cache."
                )
            fresh = await self._fetch([text for _index, text in missing])
            for (index, text), vector in zip(missing, fresh, strict=True):
                self._path(text).write_text(json.dumps(vector), encoding="utf-8")
                vectors[index] = vector
                self.misses += 1

        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)]
        )

    async def _fetch(self, texts: list[str]) -> list[list[float]]:
        from aix.core.config import config as app_config

        if self._client is None:
            self._client = app_config.openai.get_async_client()
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        if not self.one_per_request:
            response = await self._client.embeddings.create(model=self.model, input=texts, **extra)
            return [list(item.embedding) for item in sorted(response.data, key=lambda d: d.index)]

        # Some models reach a zero-data-retention endpoint only one input at a
        # time: with ZDR enforced on the account, gemini-embedding-2's only
        # batch-capable endpoint is excluded (2026-09-25). Concurrent singles.
        gate = asyncio.Semaphore(8)

        async def one(text: str) -> list[float]:
            async with gate:
                response = await self._client.embeddings.create(model=self.model, input=[text], **extra)
                return list(response.data[0].embedding)

        return list(await asyncio.gather(*(one(text) for text in texts)))


# ---------------------------------------------------------------------------
# A throwaway index over the corpus
# ---------------------------------------------------------------------------


@dataclass
class Indexed:
    """The corpus, indexed: what each document became and where it lives."""

    session_maker: Any
    #: Kept so the caller can dispose it: on Windows the temporary directory
    #: cannot be removed while SQLite still holds the file open.
    engine: Any = None
    version_ids: list[uuid.UUID] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)  # version id -> file name
    chunks_per_document: dict[str, int] = field(default_factory=dict)


async def build_index(corpus: Path, embedder: CachedEmbeddings, database: Path) -> Indexed:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from aix.webui.auth.models import User
    from aix.webui.db import Base
    from aix.webui.materials.chunking import CHUNKER_VERSION, chunk_document
    from aix.webui.materials.embedding import embed_texts
    from aix.webui.materials.models import AssetBlob, Material, MaterialVersion
    from aix.webui.materials.parsing import check_upload, parse_document
    from aix.webui.materials.repository import ChunkRepository

    engine = create_async_engine(f"sqlite+aiosqlite:///{database.as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    indexed = Indexed(session_maker=maker, engine=engine)

    async with maker() as session:
        session.add(
            User(
                id=OWNER,
                email="eval@scuola.it",
                hashed_password="x",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await session.commit()

    documents = sorted(p for p in corpus.iterdir() if p.is_file())
    if not documents:
        raise SystemExit(f"No documents in {corpus}. See the header of cases.yaml.")

    for path in documents:
        content = path.read_bytes()
        accepted = check_upload(path.name, content)
        ir = parse_document(content, accepted.media_type)
        chunks = chunk_document(ir)
        vectors = await embed_texts([c.text for c in chunks], client=embedder)

        async with maker() as session:
            blob = AssetBlob(
                owner_id=OWNER,
                storage_provider="local",
                object_key=f"eval/{path.name}",
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                verified_media_type=accepted.media_type,
            )
            material = Material(owner_id=OWNER, display_name=path.name)
            session.add_all([blob, material])
            await session.flush()

            version = MaterialVersion(
                material_id=material.id,
                version_number=1,
                blob_id=blob.id,
                source_sha256=blob.sha256,
                media_type=accepted.media_type,
                created_by=OWNER,
                status="ready",
            )
            session.add(version)
            await session.flush()

            await ChunkRepository(session).replace_for_version(
                owner_id=OWNER,
                material_version_id=version.id,
                chunker_version=CHUNKER_VERSION,
                embedding_model=embedder.model,
                chunks=list(zip(chunks, vectors, strict=True)),
            )
            await session.commit()

            indexed.version_ids.append(version.id)
            indexed.names[str(version.id)] = path.name
            indexed.chunks_per_document[path.name] = len(chunks)

    return indexed


# ---------------------------------------------------------------------------
# Scoring one case
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case_id: str
    recall: float
    #: Recall in the first three passages. With a small corpus, recall@8 is 1.00
    #: for everything and measures nothing; @3 is where a ranking change shows.
    recall_at_3: float
    mrr: float
    boilerplate: float
    missing_documents: list[str]
    seconds: float
    found: list[tuple[str, Optional[int], float]]  # document, page, score
    expected: set[tuple[str, int]]


def _pairs(entries: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (entry["document"], int(page)) for entry in entries or [] for page in entry.get("pages", [])
    }


async def run_case(
    case: dict[str, Any],
    indexed: Indexed,
    embedder: CachedEmbeddings,
    forbidden: set[tuple[str, int]],
    limit: int,
) -> CaseResult:
    from aix.webui.materials.retrieval import PrivateMaterialRetriever

    expected = _pairs(case.get("expect", []))

    async with indexed.session_maker() as session:
        started = time.perf_counter()
        items = await PrivateMaterialRetriever(session).search(
            case["query"],
            owner_id=OWNER,
            version_ids=indexed.version_ids,
            limit=limit,
            client=embedder,
        )
        seconds = time.perf_counter() - started

    found = [
        (indexed.names.get(item.source_id, item.source_id), item.locator.get("page"), item.score)
        for item in items
    ]
    found_pairs = [(name, page) for name, page, _score in found if page is not None]

    hits = expected & set(found_pairs)
    recall = len(hits) / len(expected) if expected else 1.0
    recall_at_3 = len(expected & set(found_pairs[:3])) / len(expected) if expected else 1.0

    mrr = 0.0
    for rank, pair in enumerate(found_pairs, start=1):
        if pair in expected:
            mrr = 1.0 / rank
            break

    top5 = found_pairs[:5]
    boilerplate = (sum(1 for pair in top5 if pair in forbidden) / len(top5)) if top5 else 0.0

    expected_documents = {document for document, _page in expected}
    seen_documents = {document for document, _page, _score in found}
    missing_documents = sorted(expected_documents - seen_documents)

    return CaseResult(
        case_id=case["id"],
        recall=recall,
        recall_at_3=recall_at_3,
        mrr=mrr,
        boilerplate=boilerplate,
        missing_documents=missing_documents,
        seconds=seconds,
        found=found,
        expected=expected,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def render(
    results: list[CaseResult], indexed: Indexed, embedder: CachedEmbeddings, limit: int
) -> str:
    from aix.webui.materials.chunking import CHUNKER_VERSION

    lines = [
        f"# Private RAG — retrieval on {len(results)} golden cases",
        "",
        f"chunker `{CHUNKER_VERSION}` · embeddings `{embedder.model}` · top-k {limit} · "
        f"cache {embedder.hits} hit / {embedder.misses} miss",
        "",
        "| Case | recall@3 | recall@k | MRR | boilerplate@5 | documents missing | s |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.case_id} | {result.recall_at_3:.2f} | {result.recall:.2f} | {result.mrr:.2f} "
            f"| {result.boilerplate:.2f} | {', '.join(result.missing_documents) or '—'} "
            f"| {result.seconds:.2f} |"
        )

    recalls = [r.recall for r in results]
    recalls_at_3 = [r.recall_at_3 for r in results]
    mrrs = [r.mrr for r in results]
    boilerplate = [r.boilerplate for r in results]
    latencies = sorted(r.seconds for r in results)

    lines += [
        "",
        "## Summary",
        "",
        f"- **recall@3**: mean {statistics.mean(recalls_at_3):.2f}",
        f"- **recall@{limit}**: mean {statistics.mean(recalls):.2f}, "
        f"worst {min(recalls):.2f} ({min(results, key=lambda r: r.recall).case_id})",
        f"- **MRR**: mean {statistics.mean(mrrs):.2f}",
        f"- **boilerplate@5**: mean {statistics.mean(boilerplate):.2f}, "
        f"cases affected {sum(1 for b in boilerplate if b > 0)}",
        f"- **cases missing a document**: {sum(1 for r in results if r.missing_documents)}",
        f"- **search time**: median {statistics.median(latencies):.2f}s, "
        f"slowest {latencies[-1]:.2f}s",
        "",
        "## Corpus",
        "",
        "| Document | chunks |",
        "| --- | ---: |",
    ]
    for name, count in sorted(indexed.chunks_per_document.items()):
        lines.append(f"| {name} | {count} |")

    return "\n".join(lines) + "\n"


def render_passages(result: CaseResult) -> str:
    lines = [f"### {result.case_id}", f"expected: {sorted(result.expected)}", ""]
    for rank, (document, page, score) in enumerate(result.found, start=1):
        mark = "✓" if (document, page) in result.expected else " "
        lines.append(f"  {mark} {rank:>2}. {score:.3f}  {document} p{page}")
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> int:
    import yaml

    from aix.webui.materials.embedding import embedding_model

    spec = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))
    cases = spec["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] in args.case]
        if not cases:
            raise SystemExit(f"No case matches {args.case}")

    forbidden = _pairs(spec.get("forbid_always", []))
    corpus = REPO_ROOT / spec.get("corpus", "data/eval/corpus/rag")
    embedder = CachedEmbeddings(
        embedding_model(),
        offline=args.offline,
        dimensions=args.dimensions,
        one_per_request=args.one_per_request,
    )

    import tempfile

    # ``ignore_cleanup_errors`` and the dispose below are both for Windows: the
    # directory cannot be removed while SQLite still holds the file open, and a
    # benchmark that ends in a traceback after printing nothing is useless.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        indexed = await build_index(corpus, embedder, Path(tmp) / "eval.db")
        try:
            results = [
                await run_case(case, indexed, embedder, forbidden, args.limit) for case in cases
            ]
            report = render(results, indexed, embedder, args.limit)
            if args.verbose:
                report += "\n" + "\n".join(render_passages(r) for r in results)
        finally:
            await indexed.engine.dispose()

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Written {args.out}")
    else:
        sys.stdout.write(report)

    if args.check:
        mean_recall = statistics.mean(r.recall for r in results)
        worst_boilerplate = max(r.boilerplate for r in results)
        failures = []
        if mean_recall < MIN_MEAN_RECALL:
            failures.append(f"mean recall {mean_recall:.2f} < {MIN_MEAN_RECALL}")
        if worst_boilerplate > MAX_BOILERPLATE:
            failures.append(f"boilerplate in the top 5 of {worst_boilerplate:.0%} of a case")
        if failures:
            print("\nFAIL: " + "; ".join(failures), file=sys.stderr)
            return 1
        print("\nOK: thresholds met")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=8, help="passages per case (default: top-k 8)")
    parser.add_argument("--case", action="append", help="run only this case id (repeatable)")
    parser.add_argument(
        "--offline", action="store_true", help="cache only, never call the provider"
    )
    parser.add_argument("--check", action="store_true", help="exit 1 when below the thresholds")
    parser.add_argument(
        "--dimensions", type=int, help="ask the embedding model for vectors of this size"
    )
    parser.add_argument(
        "--one-per-request", action="store_true", help="embed one text per request (for ZDR-only routes)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="list the passages found")
    parser.add_argument("--out", type=Path, help="write the report to a file")
    args = parser.parse_args(argv)

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
