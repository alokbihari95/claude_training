"""Graph RAG ingestion & indexing: the spec's 6 steps.

1 Data format & ingestion      loaders (PDF / DOCX / JSON-tabular / MD) with provenance metadata
2 Semantic & entity chunking   600/60-token splitter + LLM-guided entity-relation extraction
3 Sub-word tokenization        bge-large tokenizer for chunk length; entity normalisation / alias merge
4 Multi-modal embedding        bge-large-en-v1.5 (1024-d) chunks + Node2Vec node embeddings
5 Dual indexing                Chroma HNSW (vectors) + graph adjacency index with edge->chunk back-refs
6 Hybrid store                 Chroma (persisted) + NetworkX (persisted); BM25 built at load

CLI:  python -m src.ingest [--rebuild] [--no-graph] [--chunk-size N --chunk-overlap N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Literal

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

import config
from src.graph_store import ChunkRef, EntityResolver, ExtractedTriple, GraphStore
from src.llm import count_tokens, get_embeddings, get_llm, get_tokenizer
from src.retrieval import COLLECTION, reset_index

EXTRACTION_VERSION = "v2"
FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)
SEPARATORS = ["\n## ", "\nSection ", "\nRule ", "\nClause ", "\nPara ", "\n\n", "\n", ". ", " ", ""]


# --------------------------------------------------------------------------- step 1: loaders
def _manifest() -> dict:
    p = config.CORPUS_DIR / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _base_meta(path: Path, practice_area: str | None, manifest: dict) -> dict:
    m = dict(manifest.get(path.name, {}))
    if not m:
        m = {"doc_id": path.stem, "title": path.stem.replace("_", " "), "practice_area": practice_area or config.ALL_AREAS,
             "doc_type": "uploaded", "effective_date": ""}
    m["source"] = path.name
    m["format"] = path.suffix.lstrip(".").lower()
    return m


def load_file(path: Path, practice_area: str | None = None, manifest: dict | None = None) -> list[Document]:
    manifest = _manifest() if manifest is None else manifest
    meta = _base_meta(path, practice_area, manifest)
    ext = path.suffix.lower()
    docs: list[Document] = []
    if ext == ".pdf":
        try:  # spec: UnstructuredPDFLoader; falls back to PyPDF when `unstructured` is unavailable (no py3.14 wheel)
            from langchain_community.document_loaders import UnstructuredPDFLoader

            docs = UnstructuredPDFLoader(str(path), mode="single").load()
            meta["loader"] = "UnstructuredPDFLoader"
        except Exception:  # noqa: BLE001
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(str(path)).load()
            meta["loader"] = "PyPDFLoader"
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
        meta["loader"] = "Docx2txtLoader"
    elif ext == ".json":
        payload = json.loads(path.read_text())
        meta["loader"] = "JSONTableLoader"
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if rows:  # structured tax table: one row = one document
            note = payload.get("note", "")
            for i, row in enumerate(rows):
                text = f"{note}\n" + "; ".join(f"{k}: {v}" for k, v in row.items())
                docs.append(Document(page_content=text, metadata={"row": i}))
        else:
            docs = [Document(page_content=json.dumps(payload, indent=1), metadata={})]
    elif ext in {".md", ".txt"}:
        text = path.read_text()
        text = FRONT_MATTER.sub("", text, count=1)
        docs = [Document(page_content=text, metadata={})]
        meta["loader"] = "TextLoader"
    else:
        raise ValueError(f"unsupported file type: {path.name}")
    out = []
    for d in docs:
        md = {**meta, **d.metadata}
        md["page"] = (d.metadata.get("page", 0) + 1) if ext == ".pdf" and "page" in d.metadata else d.metadata.get("page")
        md.pop("page_label", None)
        out.append(Document(page_content=d.page_content.strip(), metadata={k: v for k, v in md.items() if v is not None}))
    return [d for d in out if d.page_content]


# --------------------------------------------------------------------------- steps 2-3: chunking
def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        get_tokenizer(), chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS
    )


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks, counters = [], {}
    for d in docs:
        # JSON table rows are already atomic facts: never split or merge them
        pieces = [d.page_content] if d.metadata.get("loader") == "JSONTableLoader" else splitter.split_text(d.page_content)
        for piece in pieces:
            doc_id = d.metadata["doc_id"]
            idx = counters.get(doc_id, 0)
            counters[doc_id] = idx + 1
            title = d.metadata.get("title", doc_id)
            text = f"{title}\n{piece}"   # title prefix keeps clause context; identical for every engine
            chunks.append({
                "chunk_id": f"{doc_id}::c{idx}",
                "text": text,
                "metadata": {**d.metadata, "chunk_index": idx, "n_tokens": count_tokens(text)},
            })
    return chunks


# --------------------------------------------------------------------------- step 2: entity-relation extraction
class _Entity(BaseModel):
    name: str
    type: Literal["Regulator", "Provision", "Instrument", "Organization", "Obligation", "Penalty",
                  "Control", "Concept", "Value", "Date"] = "Concept"


class _Relation(BaseModel):
    subject: str
    relation: str = Field(description="short lowercase verb phrase, e.g. 'clarifies', 'has penalty', 'maps to'")
    object: str


class Extraction(BaseModel):
    entities: list[_Entity]
    relations: list[_Relation]


EXTRACT_SYSTEM = """You extract a regulatory/contract knowledge graph from one passage.
Rules:
- Entities: regulators, provisions (sections, rules, paragraphs, clauses), instruments (notifications, circulars, agreements, forms), organizations, obligations, penalties, controls, key concepts, and literal Values/Dates (thresholds, rates, amounts, periods).
- Use the FULL canonical name as written in the passage. Resolve defined terms and pronouns to the real entity (for example 'the Company' -> the company's full name, 'the Buyer' -> the buyer's full name, 'the Foundation' -> its full name).
- Use the common abbreviation for well-known regulators (CBDT, CBIC, RBI, SEBI, FIU-IND).
- Relations are directed (subject -> object) with a short lowercase verb phrase. Capture cross-references between documents (amends, clarifies, refers to, maps to, requires, tests, covers) and facts (has rate, has threshold, has penalty, effective from, applies to).
- Attach each Value/Date to the entity it describes via a relation, e.g. (Section 94Z, has rate, 10%).
- Only extract what the passage states. Do not invent."""


def _cache_load() -> dict:
    return json.loads(config.EXTRACTION_CACHE.read_text()) if config.EXTRACTION_CACHE.exists() else {}


def extract_all(chunks: list[dict], log: Callable[[str], None] = print) -> dict[str, Extraction]:
    """LLM-guided entity-relation extraction, cached by (model, version, text) so re-runs cost nothing."""
    cache = _cache_load()
    keys = {c["chunk_id"]: hashlib.sha1(f"{config.LLM_MODEL}|{EXTRACTION_VERSION}|{c['text']}".encode()).hexdigest() for c in chunks}
    todo = [c for c in chunks if keys[c["chunk_id"]] not in cache]
    log(f"  extraction: {len(chunks) - len(todo)} cached, {len(todo)} to run")
    if todo:
        llm = get_llm(max_tokens=8000).with_structured_output(Extraction)
        msgs = [[("system", EXTRACT_SYSTEM), ("human", f"Document: {c['metadata'].get('title')}\n\nPassage:\n{c['text']}")] for c in todo]
        results = llm.batch(msgs, config={"max_concurrency": 6}, return_exceptions=True)
        for c, r in zip(todo, results):
            if isinstance(r, Exception) or r is None:
                log(f"  ! extraction failed for {c['chunk_id']}: {r}")
                continue
            cache[keys[c["chunk_id"]]] = r.model_dump()
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        config.EXTRACTION_CACHE.write_text(json.dumps(cache))
    return {c["chunk_id"]: Extraction(**cache[keys[c["chunk_id"]]]) for c in chunks if keys[c["chunk_id"]] in cache}


# --------------------------------------------------------------------------- steps 4-6: embed, index, graph
def _embed_fn(texts: list[str]):
    return get_embeddings().embed_documents(texts)


def _persist_chunks(chunks: list[dict]) -> None:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (config.INDEX_DIR / "chunks.json").write_text(json.dumps(chunks))


def _load_chunks() -> list[dict]:
    p = config.INDEX_DIR / "chunks.json"
    return json.loads(p.read_text()) if p.exists() else []


def _add_to_chroma(col, chunks: list[dict]) -> None:
    if not chunks:
        return
    vecs = get_embeddings().embed_documents([c["text"] for c in chunks])
    col.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vecs,
        documents=[c["text"] for c in chunks],
        metadatas=[{k: v for k, v in c["metadata"].items() if isinstance(v, (str, int, float, bool))} for c in chunks],
    )


def _update_graph(gs: GraphStore, chunks: list[dict], ext: dict[str, Extraction]) -> None:
    resolver = EntityResolver(gs.G, embed_fn=_embed_fn)
    for c in chunks:
        e = ext.get(c["chunk_id"])
        if not e:
            continue
        types = {x.name: x.type for x in e.entities}
        ents = [(x.name, x.type) for x in e.entities]
        triples = [ExtractedTriple(r.subject, r.relation, r.object, types.get(r.subject, "Concept"), types.get(r.object, "Concept")) for r in e.relations]
        gs.add_extraction(ChunkRef(c["chunk_id"], c["metadata"]["doc_id"], c["metadata"]["practice_area"]), ents, triples, resolver)
    gs.compute_node2vec()
    gs.save()


def ingest_paths(
    paths: list[Path],
    practice_area: str | None = None,
    chunk_size: int = config.DEFAULTS.chunk_size,
    chunk_overlap: int = config.DEFAULTS.chunk_overlap,
    build_graph: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Incrementally ingest files (steps 1-6). Existing chunks of the same doc_id are replaced."""
    t0 = time.time()
    manifest = _manifest()
    log("[1/6] loading documents")
    docs = [d for p in paths for d in load_file(Path(p), practice_area, manifest)]
    log(f"      {len(docs)} document units from {len(paths)} files; loaders: {sorted({d.metadata['loader'] for d in docs})}")
    log(f"[2-3/6] chunking at {chunk_size}/{chunk_overlap} tokens (bge tokenizer)")
    new_chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    over = [c["chunk_id"] for c in new_chunks if c["metadata"]["n_tokens"] > config.EMBEDDING_MAX_TOKENS]
    if over:
        log(f"      WARNING: {len(over)} chunks exceed the {config.EMBEDDING_MAX_TOKENS}-token bge limit and will be truncated at embedding: {over[:5]}")
    replaced = {c["metadata"]["doc_id"] for c in new_chunks}
    kept = [c for c in _load_chunks() if c["metadata"]["doc_id"] not in replaced]
    all_chunks = kept + new_chunks
    _persist_chunks(all_chunks)

    log("[4-5/6] embedding chunks (bge-large-en-v1.5) and updating HNSW index")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    stale = [i for i in col.get()["ids"] if i.split("::")[0] in replaced]
    if stale:
        col.delete(ids=stale)
    _add_to_chroma(col, new_chunks)

    stats = {}
    if build_graph:
        log("[2,4-6/6] entity-relation extraction, entity resolution, Node2Vec, adjacency index")
        gs = GraphStore.load()
        if replaced & {d for _, x in gs.G.nodes(data=True) for d in x.get("docs", [])}:
            gs = GraphStore()   # re-ingesting an existing doc: rebuild the graph from all chunks for consistency
            _update_graph(gs, all_chunks, extract_all(all_chunks, log))
        else:
            _update_graph(gs, new_chunks, extract_all(new_chunks, log))
        stats = gs.stats()
        log(f"      graph: {stats}")
    reset_index()
    out = {"chunks": len(new_chunks), "total_chunks": len(all_chunks), "graph": stats, "seconds": round(time.time() - t0, 1)}
    log(f"done: {out}")
    return out


def build_all(rebuild: bool = False, build_graph: bool = True, chunk_size: int = config.DEFAULTS.chunk_size,
              chunk_overlap: int = config.DEFAULTS.chunk_overlap, log: Callable[[str], None] = print) -> dict:
    if rebuild and config.INDEX_DIR.exists():
        for p in config.INDEX_DIR.iterdir():
            if p.name == EXTRACTION_CACHE_NAME:   # keep the extraction cache: re-chunking must not re-pay for the LLM
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    files = sorted(p for p in config.CORPUS_DIR.iterdir() if p.suffix.lower() in {".md", ".pdf", ".docx", ".json"} and p.name != "manifest.json")
    files += sorted(p for p in config.UPLOAD_DIR.glob("*") if p.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}) if config.UPLOAD_DIR.exists() else []
    return ingest_paths(files, None, chunk_size, chunk_overlap, build_graph, log)


EXTRACTION_CACHE_NAME = config.EXTRACTION_CACHE.name

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=config.DEFAULTS.chunk_size)
    ap.add_argument("--chunk-overlap", type=int, default=config.DEFAULTS.chunk_overlap)
    a = ap.parse_args()
    build_all(a.rebuild, not a.no_graph, a.chunk_size, a.chunk_overlap)
