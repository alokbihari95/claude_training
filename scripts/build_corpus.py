"""Render data/corpus_src/* into data/corpus/ as real PDF / DOCX / MD / JSON files.

Each source file carries YAML-ish front matter (doc_id, title, practice_area, doc_type, format,
effective_date). The binary formats lose that header, so provenance is also written to
data/corpus/manifest.json, which the ingestion pipeline reads.

Usage: python scripts/build_corpus.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CORPUS_DIR, DATA_DIR  # noqa: E402

SRC = DATA_DIR / "corpus_src"
FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def parse_front_matter(text: str) -> tuple[dict, str]:
    m = FM.match(text)
    if not m:
        raise ValueError("missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, m.group(2).strip() + "\n"


def latin1(s: str) -> str:
    return s.encode("latin-1", "replace").decode("latin-1")


def inline(s: str) -> str:
    return s.replace("**", "")


def write_docx(meta: dict, body: str, out: Path) -> None:
    from docx import Document

    doc = Document()
    doc.core_properties.title = meta["title"]
    for line in body.splitlines():
        if not line.strip():
            continue
        if line.startswith("## "):
            doc.add_heading(inline(line[3:]), level=2)
        elif line.startswith("# "):
            doc.add_heading(inline(line[2:]), level=1)
        elif line.lstrip().startswith("- "):
            doc.add_paragraph(inline(line.lstrip()[2:]), style="List Bullet")
        else:
            doc.add_paragraph(inline(line))
    doc.save(out)


def write_pdf(meta: dict, body: str, out: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_title(meta["title"])
    for line in body.splitlines():
        if not line.strip():
            pdf.ln(2)
            continue
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            text = line[2:]
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 11)
            text = line[3:]
        else:
            pdf.set_font("Helvetica", "", 10)
            text = line
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, latin1(inline(text)))
    pdf.output(str(out))


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for src in sorted(SRC.iterdir()):
        if src.suffix == ".json":
            payload = json.loads(src.read_text())
            meta = {k: payload[k] for k in ("doc_id", "title", "practice_area", "doc_type", "format", "effective_date")}
            out = CORPUS_DIR / f"{meta['doc_id']}.json"
            out.write_text(json.dumps(payload, indent=2))
        else:
            meta, body = parse_front_matter(src.read_text())
            fmt = meta["format"]
            out = CORPUS_DIR / f"{meta['doc_id']}.{fmt}"
            if fmt == "docx":
                write_docx(meta, body, out)
            elif fmt == "pdf":
                write_pdf(meta, body, out)
            else:
                out.write_text(body)
        manifest[out.name] = meta
        print(f"built {out.name:14s} {meta['practice_area']:6s} {meta['doc_type']}")
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} documents -> {CORPUS_DIR}")


if __name__ == "__main__":
    main()
