# Standard RAG vs Multi-HyDE RAG vs Graph RAG: PwC India Advisory benchmark

LangChain + LangGraph implementation of the architecture spec. Three engines answer the same questions over the same
chunks, embeddings and HNSW index, with one shared CO-STAR prompt, so retrieval strategy is the only variable.

> **All documents are synthetic.** Section numbers, rates, notifications and the Meridian/Axiom deal are fictional.
> 15 hand-written core documents (Tax / GRC / Deals, in PDF, DOCX, JSON-table and MD) + 29 generated near-miss
> *distractors* (`DIST-*`) that make retrieval non-trivial.

## Setup
```bash
python -m venv .venv --system-site-packages && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install -r requirements.txt                                      # drop `unstructured` if it will not build (PyPDF fallback)
export OPENROUTER_API_KEY=...                                        # generation, HyDE, extraction, judge
python scripts/build_corpus.py        # data/corpus_src -> data/corpus (real PDF/DOCX/MD/JSON + manifest)
python -m src.ingest --rebuild        # the 6-step pipeline (first run downloads bge-large ~1.3 GB)
python app.py                         # http://127.0.0.1:7870  (PORT env var overrides)
python -m src.evaluate                # full benchmark -> results/benchmark.json + .csv
pytest tests/                         # unit tests (no LLM calls)
```
`scripts/generate_distractors.py` regenerates the distractors (LLM; output is not deterministic).

## Layout
| Path | Role |
|---|---|
| `config.py` | models, the 4 spec hyperparameters, practice areas + guardrails |
| `src/prompts.py` | CO-STAR base prompt + per-practice-area guardrails (shared by all engines) |
| `src/ingest.py` | spec steps 1-6: loaders, 600/60-token chunking, LLM entity-relation extraction, embeddings, dual index |
| `src/graph_store.py`, `src/node2vec.py` | NetworkX graph, entity resolution (seed aliases, fuzzy, embedding), k-hop expansion, Node2Vec (torch) |
| `src/retrieval.py` | Chroma HNSW, BM25, RRF fusion (`1/(c+rank)`, c=60), bge reranker |
| `src/engines.py` | LangGraph graphs: Standard, Multi-HyDE (N=3), Graph RAG (cyclic k-hop loop) |
| `src/metrics.py`, `src/evaluate.py` | the 4 KPIs, benchmark runner, comparison against the spec's expectations |
| `app.py`, `ui/` | 5-tab Gradio app |
| `data/gold_qa.json` | 24 questions (8 simple, 8 vocab-mismatch, 8 multi-hop) with atomic facts + gold docs |
| `data/spec_expectations.json` | the spec's expected comparison matrix: **hypotheses, never shown as results** |

## KPIs
RCR (judge: gold atomic facts present in retrieved chunks) · Faithfulness (supported claims / claims; hallucination = 1 - faithfulness)
· Entity-Relation Resolution accuracy (graph vs `gold_entities.json`; n/a for Standard/HyDE) · end-to-end latency with
retrieval / graph / LLM breakdown. Answer correctness, gold-document recall and tokens are reported as supporting metrics.

## Known deviations and caveats (please read)
- **Chunk size 600 vs bge-large's 512-token limit.** The spec asks for 600-token chunks and bge-large-en-v1.5 truncates input at 512. Ingestion warns if a chunk exceeds 512. Current documents are short, so none do.
- **UnstructuredPDFLoader** has no Python 3.14 wheel here; PDFs load via PyPDFLoader (same metadata). **gensim** likewise: Node2Vec skip-gram is trained in torch.
- **Neo4j and Qdrant** are not built; NetworkX and Chroma sit behind small interfaces (`GraphStore`, `RAGIndex`).
- **Latency** depends on the OpenRouter model, network and (here) a shared 4-core CPU running a CPU reranker. The spec's ~450/850/1,200 ms are not reproducible in this setup; only the *ordering* is compared.
- **Spec table pairing.** The spec's latency and hallucination cells were garbled in extraction; I read them column-wise (Standard 450, HyDE 1,200, Graph 850). Confirm in `data/spec_expectations.json`.
- **Single-user local app:** cross-tab settings live in one module-level dict, not per-session.
- LLM-as-judge scores (RCR, faithfulness) come from one judge model at temperature 0; they are indicative, not ground truth.
