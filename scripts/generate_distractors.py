"""Generate 30 near-miss DISTRACTOR documents (same domains and vocabulary, different provisions/entities).

Why: with only 15 hand-written documents, top-5 retrieval covers a third of the corpus and every engine saturates.
Distractors make retrieval non-trivial. They are forbidden from restating any gold fact (checked after generation).

Usage: python scripts/generate_distractors.py   (then: python scripts/build_corpus.py && python -m src.ingest --rebuild)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.llm import get_llm  # noqa: E402

TOPICS = [
    # tax
    ("tax", "notification", "CBDT Notification: Section 94Y withholding on royalty payments to non-residents (rate 12%, threshold Rs 25 lakh)"),
    ("tax", "circular", "CBDT Circular: Section 94X advance ruling application procedure and timelines"),
    ("tax", "circular", "CBDT Circular: transfer pricing safe-harbour margins for IT-enabled services"),
    ("tax", "notification", "CBDT Notification: equalisation levy on digital advertising services"),
    ("tax", "circular", "CBIC Circular: GST composition scheme eligibility for small manufacturers"),
    ("tax", "circular", "CBIC Circular: GST refund of unutilised credit under inverted duty structure"),
    ("tax", "ruling", "55th GST Council ruling: GST on renting of residential property to registered persons"),
    ("tax", "circular", "CBIC Circular: customs valuation of related-party imports"),
    ("tax", "notification", "CBDT Notification: presumptive taxation scheme for small professionals"),
    ("tax", "ruling", "56th GST Council ruling: GST rate on electric-vehicle charging services"),
    # grc
    ("grc", "master_direction", "RBI Master Direction on digital lending: key fact statement and cooling-off period"),
    ("grc", "master_direction", "RBI Master Direction on outsourcing of IT services by regulated entities"),
    ("grc", "circular", "SEBI circular: LODR related-party transaction approval thresholds for listed entities"),
    ("grc", "circular", "SEBI circular: cyber resilience framework for stock brokers"),
    ("grc", "policy", "ISO 31000 risk appetite statement for Orion Insurance Ltd"),
    ("grc", "sop", "Internal audit SOP-IA-P2P-01: procure-to-pay controls testing at Orion Capital Ltd"),
    ("grc", "circular", "RBI circular: credit information reporting by lenders to credit bureaus"),
    ("grc", "guidelines", "Insurance regulator guidelines on corporate governance and board committees for insurers"),
    ("grc", "control_register", "Orion Capital Ltd: internal financial controls register mapped to Companies Act requirements"),
    ("grc", "circular", "SEBI circular: registration framework for ESG rating providers"),
    # deals
    ("deals", "spa", "Share Purchase Agreement between Kestrel Capital and Orion Textiles Ltd (indemnity cap 15%, escrow 12% for 24 months, working-capital adjustment)"),
    ("deals", "statute_extract", "Companies Act 2013 Section 188: related-party transactions approvals"),
    ("deals", "statute_extract", "Companies Act 2013 Section 230: schemes of arrangement and merger approvals"),
    ("deals", "rules", "FEMA overseas direct investment rules: eligibility and reporting by Indian parties"),
    ("deals", "dd_memo", "Due diligence summary Project Orion: GST and labour-law findings for Orion Textiles Ltd"),
    ("deals", "agreement", "Shareholders agreement for Helix Pharma Ltd: drag-along, tag-along and board reserved matters"),
    ("deals", "rules", "SEBI takeover code: open offer triggers and pricing for listed targets"),
    ("deals", "rules", "Competition regulator: combination notification thresholds and review timelines"),
    ("deals", "term_sheet", "Term sheet for minority investment in Zephyr Logistics: anti-dilution and liquidation preference"),
    ("deals", "circular", "State stamp duty circular on share transfer instruments and agreements to sell"),
]
FORBIDDEN = ["Section 94Z", "271Z", "Meridian", "Axiom", "Rule 39A", "ISD", "IRN", "FC-TRS", "Harbor Trust", "SYN-CBDT/14/2025", "SYN-CBDT-C/07/2025",
             "SYN-CBIC/203/2025", "SYN-RBI/KYC/2025", "SOP-IA-KYC-03", "IC-KYC-07"]
SYSTEM = ("You write SYNTHETIC test documents that imitate Indian regulatory, compliance and M&A texts for a retrieval benchmark. "
          "Everything is fictional. Output only the document body in markdown.")
PROMPT = """Write one synthetic document, 180-260 words, on this topic:
{topic}

Requirements:
- First line exactly: "SYNTHETIC DOCUMENT FOR BENCHMARKING. NOT OFFICIAL TEXT. All figures are fictional."
- Then a '#' title with a fictional identifier like SYN-XXXX/{n}/2025, and 3-5 '##' numbered sections with concrete thresholds, rates, periods, penalties and dates (invent them).
- Use the same register and vocabulary as real Indian advisory documents (withholding, indemnity, escrow, controls, circular, notification).
- Do NOT mention any of: {forbidden}. Do not define beneficial-owner thresholds, suspicious-transaction report deadlines, KYC update frequencies, record-retention periods, e-invoicing thresholds, or valuation-report validity."""


def main() -> None:
    llm = get_llm(max_tokens=3000, temperature=0.7)
    msgs = [[("system", SYSTEM), ("human", PROMPT.format(topic=t, n=i + 1, forbidden=", ".join(FORBIDDEN)))] for i, (_, _, t) in enumerate(TOPICS)]
    outs = llm.batch(msgs, config={"max_concurrency": 6}, return_exceptions=True)
    src = config.DATA_DIR / "corpus_src"
    bad = 0
    for i, ((area, dtype, topic), o) in enumerate(zip(TOPICS, outs), 1):
        if isinstance(o, Exception):
            print(f"! DIST-{i:02d} failed: {o}")
            bad += 1
            continue
        body = str(o.content).strip()
        hits = [f for f in FORBIDDEN if f.lower() in body.lower()]
        if hits:
            print(f"! DIST-{i:02d} mentions forbidden {hits}; skipped")
            bad += 1
            continue
        title = topic.split(":")[0] if ":" in topic and len(topic.split(":")[0]) > 12 else topic[:90]
        fm = (f"---\ndoc_id: DIST-{i:02d}\ntitle: {topic[:110]}\npractice_area: {area}\ndoc_type: {dtype}\nformat: md\n"
              f"effective_date: 2025-01-01\nrole: distractor\n---\n")
        (src / f"DIST-{i:02d}.md").write_text(fm + body + "\n")
    print(f"wrote {len(TOPICS) - bad} distractors, {bad} problems")


if __name__ == "__main__":
    main()
