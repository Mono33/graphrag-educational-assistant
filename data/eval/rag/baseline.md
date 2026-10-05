# Private RAG — retrieval on 12 golden cases

Regenerate with `python scripts/eval/run_private_rag.py --offline --out data/eval/rag/baseline.md`.
This file is the recorded state of the ranking: a change to retrieval is
accompanied by a new run of this table, or it is not a change worth keeping.

## What has been measured, in order

| # | Change | recall@3 | MRR | boilerplate@5 | kept |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | vector only (2026-09-18) | 0.88 | 0.78 | 0.22 · 8 cases | — |
| 1 | at most 2 passages per page | 0.88 | 0.78 | 0.23 · 9 cases | **no** |
| 2 | hybrid: BM25 fused with the vector ranking by RRF | 0.92 | 0.86 | 0.07 · 4 cases | yes |
| 3 | share cap 0.5 / 0.6 / 0.75 × candidate pool ×3 / ×5 / ×8 | 0.92 | 0.86 | 0.07 · 4 cases | no change |

Change 1 was the intuition that the bibliography wins because one page supplies
several passages. The measurement said otherwise — capping a page simply let a
different boilerplate page in — so it was reverted.

Change 2 is the §8.5 hybrid. It is the only change so far that improved every
number at once, and it did so without costing latency.

Change 3 found the ranking insensitive to all three constants on this corpus:
two documents and 38 passages are too few for a share cap to bind. They stay at
their current values, and the sweep is worth repeating when the corpus grows.

**Still open.** Four cases keep one boilerplate passage in their top five:
`Cold-War.pdf` p6 (a list of Common Core standards) and p7 (the bibliography).
A shape rule was tried and rejected on the evidence: the standards page is
written in full sentences and scores 0.93 on a prose-ratio measure, above 11
genuine content passages, so no threshold separates them. A keyword rule on
"standard" is worse still — in the Italian ministerial corpus this product is
aimed at, the curriculum standards *are* the content the teacher wants cited.
What is left is semantic (a page about *teaching* the subject rather than about
the subject), which is a reranker's job — §8.5 step 4, and only if it proves
necessary once the corpus is bigger than two documents.

chunker `chunk@2` · embeddings `baai/bge-m3` · top-k 8 · cache 50 hit / 0 miss

| Case | recall@3 | recall@k | MRR | boilerplate@5 | documents missing | s |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| yalta | 1.00 | 1.00 | 1.00 | 0.20 | — | 0.02 |
| due-blocchi | 0.50 | 1.00 | 0.50 | 0.00 | — | 0.01 |
| piano-marshall | 1.00 | 1.00 | 1.00 | 0.20 | — | 0.01 |
| dottrina-truman | 1.00 | 1.00 | 1.00 | 0.20 | — | 0.01 |
| muro-di-berlino | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| nato-patto-varsavia | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| equilibrio-del-terrore | 1.00 | 1.00 | 1.00 | 0.20 | — | 0.01 |
| bandung-terzo-mondo | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| guerra-di-corea | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| crisi-di-cuba | 0.50 | 1.00 | 0.50 | 0.00 | — | 0.01 |
| fine-della-guerra-fredda | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| discorso-cortina-di-ferro | 1.00 | 1.00 | 0.33 | 0.00 | — | 0.01 |

## Summary

- **recall@3**: mean 0.92
- **recall@8**: mean 1.00, worst 1.00 (yalta)
- **MRR**: mean 0.86
- **boilerplate@5**: mean 0.07, cases affected 4
- **cases missing a document**: 0
- **search time**: median 0.01s, slowest 0.02s

## Corpus

| Document | chunks |
| --- | ---: |
| Cold-War.pdf | 29 |
| guerra-fredda-def.pdf | 9 |
