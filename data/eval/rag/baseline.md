# Private RAG — retrieval on 12 golden cases

chunker `chunk@2` · embeddings `baai/bge-m3` · top-k 8 · cache 50 hit / 0 miss

| Case | recall@3 | recall@k | MRR | boilerplate@5 | documents missing | s |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| yalta | 1.00 | 1.00 | 1.00 | 0.40 | — | 0.01 |
| due-blocchi | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| piano-marshall | 1.00 | 1.00 | 1.00 | 0.40 | — | 0.01 |
| dottrina-truman | 1.00 | 1.00 | 1.00 | 0.40 | — | 0.01 |
| muro-di-berlino | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| nato-patto-varsavia | 0.00 | 1.00 | 0.25 | 0.20 | — | 0.01 |
| equilibrio-del-terrore | 1.00 | 1.00 | 0.50 | 0.20 | — | 0.01 |
| bandung-terzo-mondo | 1.00 | 1.00 | 1.00 | 0.20 | — | 0.01 |
| guerra-di-corea | 1.00 | 1.00 | 1.00 | 0.40 | — | 0.01 |
| crisi-di-cuba | 0.50 | 1.00 | 0.33 | 0.00 | — | 0.01 |
| fine-della-guerra-fredda | 1.00 | 1.00 | 1.00 | 0.00 | — | 0.01 |
| discorso-cortina-di-ferro | 1.00 | 1.00 | 0.33 | 0.40 | — | 0.01 |

## Summary

- **recall@3**: mean 0.88
- **recall@8**: mean 1.00, worst 1.00 (yalta)
- **MRR**: mean 0.78
- **boilerplate@5**: mean 0.22, cases affected 8
- **cases missing a document**: 0
- **search time**: median 0.01s, slowest 0.01s

## Corpus

| Document | chunks |
| --- | ---: |
| Cold-War.pdf | 29 |
| guerra-fredda-def.pdf | 9 |
