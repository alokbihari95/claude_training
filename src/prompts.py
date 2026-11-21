"""Unified CO-STAR prompt framework with per-practice-area guardrails.

Every engine (Standard, HyDE, Graph) builds its generation prompt from `build_system_prompt`, so the
only experimental variable across engines is retrieval.
"""
from __future__ import annotations

import config

COSTAR = """\
# CONTEXT
You are a Senior Principal Advisory AI Assistant serving PwC India domain consultants.

# OBJECTIVE
Deliver accurate, verifiable, and cited compliance analysis from the indexed enterprise corpora ONLY.
Never use outside knowledge. Every factual claim must be supported by a retrieved passage.

# STYLE
Professional, analytical, structured, and auditable corporate advisory.

# TONE
Authoritative, objective, risk-aware, and precise.

# AUDIENCE
Partners, Enterprise Clients, Tax Controllers, Chief Risk Officers, and Audit Committees.

# RESPONSE FORMAT
Structured markdown with inline statutory/clause citations in the form [DOC-ID, Section/Para/Clause].
Cite the document ID given in each source header. End with a section titled "Gaps / Not found in corpus"
that lists anything the question asks for that the sources do not support. If the sources do not contain
the answer, say so explicitly rather than guessing.
"""

GENERIC_GUARDRAIL = (
    "Cross-domain question: apply the guardrails of every practice area that is relevant to the "
    "sources you cite."
)


def area_block(practice_area: str) -> str:
    if practice_area == config.ALL_AREAS or practice_area not in config.PRACTICE_AREAS:
        parts = [
            f"- {v['label']}: {v['guardrails']}" for v in config.PRACTICE_AREAS.values()
        ]
        return "# PRACTICE AREA GUARDRAILS (All areas)\n" + GENERIC_GUARDRAIL + "\n" + "\n".join(parts)
    a = config.PRACTICE_AREAS[practice_area]
    return (
        f"# PRACTICE AREA: {a['label']}\n"
        f"Regulatory focus: {a['focus']}\n"
        f"ENFORCED GUARDRAIL RULES: {a['guardrails']}"
    )


def build_system_prompt(practice_area: str = config.ALL_AREAS) -> str:
    return COSTAR + "\n" + area_block(practice_area)


def format_context(contexts: list[dict]) -> str:
    """Render retrieved passages with the source header the model must cite."""
    blocks = []
    for i, c in enumerate(contexts, 1):
        m = c.get("metadata", {})
        head = f"[Source {i}] DOC-ID: {m.get('doc_id', '?')} | {m.get('title', '')}"
        if m.get("page") is not None:
            head += f" | page {m['page']}"
        blocks.append(f"{head}\n{c['text']}")
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


def build_user_prompt(question: str, contexts: list[dict], extra: str = "") -> str:
    return (
        f"SOURCES\n{format_context(contexts)}\n\n"
        + (f"{extra}\n\n" if extra else "")
        + f"QUESTION\n{question}"
    )


HYDE_SYSTEM = (
    "You write hypothetical regulatory/contractual passages used ONLY to improve document search. "
    "They need not be factually correct; they must use the precise vocabulary, section-style phrasing and "
    "terminology that an official Indian tax, risk-and-compliance, or M&A document would use."
)

HYDE_ANGLES = [
    "the operative statutory or contractual text (provision wording, thresholds, rates)",
    "an official circular/clarification or guidance note explaining scope, timing and interpretation",
    "the compliance obligations, penalties and remediation steps that follow from the provision",
]
