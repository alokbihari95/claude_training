"""Build 3 sample files (PDF + 2 DOCX) for the 'Ingestion & Chat Console' tab. Output: data/samples/

They contain facts that are NOT in the main corpus, so uploading them visibly changes answers. One (the SPA amendment)
links to existing documents (Section 94Z, Registered Valuer, Meridian), so cross-document multi-hop questions work.
All content is synthetic. Usage: python scripts/build_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_corpus import write_docx, write_pdf  # noqa: E402

OUT = ROOT / "data" / "samples"

SAMPLES = [
    ("Sample_Tax_Circular_Section_94W.pdf", write_pdf, "tax", "CBDT Circular SYN-CBDT-C/11/2026 - Section 94W Director Sitting Fees", """\
SYNTHETIC DOCUMENT FOR BENCHMARKING. Fictional section numbers, rates and dates.

# CBDT Circular No. SYN-CBDT-C/11/2026 (dated 9 February 2026)

The Central Board of Direct Taxes (CBDT) clarifies the operation of Section 94W of the Income Tax Act (synthetic) on tax withheld from sitting fees paid to company directors.

## 1. Rate and threshold (prospective)
With effect from 1 April 2026, a company shall withhold tax at 8% on sitting fees and commission paid to a director in a financial year where the aggregate exceeds Rs 2 lakh. The provision applies only to payments made on or after 1 April 2026.

## 2. Independent directors
Sitting fees paid to independent directors who are non-resident are subject to the same 8% rate and the Rs 2 lakh threshold. There is no lower treaty rate unless the director furnishes a tax residency certificate by 30 April of the following year.

## 3. Reimbursement of expenses
Reimbursement of actual travel and lodging costs supported by invoices is not sitting fees and is outside Section 94W.

## 4. Default
Interest at 1.25% per month applies on tax not withheld. Penalty under Section 271W equals 50% of the tax not withheld.

## 5. Filing
Withholding under Section 94W shall be reported in Form 26W by the 15th day after the end of each quarter.
"""),
    ("Sample_GRC_Third_Party_Risk_SOP.docx", write_docx, "grc", "SOP-TPR-05 Third-Party Risk Management (ISO 31000 aligned)", """\
SYNTHETIC DOCUMENT FOR BENCHMARKING. Fictional company, controls and figures.

# SOP-TPR-05 - Third-Party Risk Management at Orion Capital Ltd

## 1. Purpose
To manage risks arising from outsourced vendors, aligned to ISO 31000, and to map each control to its regulatory obligation.

## 2. Vendor tiering
- Tier 1 (critical): vendors handling customer data or processing more than Rs 50 crore a year. Reviewed every 12 months.
- Tier 2 (important): reviewed every 24 months.
- Tier 3 (standard): reviewed every 36 months.

## 3. Controls
- Control TPR-01: Due diligence questionnaire before onboarding. Mapped to RBI outsourcing direction, Para 7.
- Control TPR-02: Right-to-audit clause in every Tier 1 contract. Mapped to RBI outsourcing direction, Para 11.
- Control TPR-03: Exit plan tested annually for Tier 1 vendors.

## 4. Incident escalation
A vendor security incident must be reported to the Chief Information Security Officer within 4 hours and to the Audit Committee within 2 working days.

## 5. Penalty exposure
Failure to maintain TPR-02 may attract a regulatory penalty of up to Rs 50 lakh per contravention. This penalty must be flagged to the Risk Committee immediately.
"""),
    ("Sample_Deal_SPA_Amendment_1_Meridian.docx", write_docx, "deals", "Amendment No. 1 to the Axiom-Meridian Share Purchase Agreement", """\
SYNTHETIC DOCUMENT FOR BENCHMARKING. Fictional parties, amounts and clauses. Extends the Project Meridian deal.

# Amendment No. 1 to the Share Purchase Agreement between Axiom Holdings Pte Ltd and the Sellers of Meridian Components Ltd

## Clause 13 - Transition Services Agreement (new)
For 6 months after Closing, Meridian Global GmbH shall provide transition services (IT, treasury and payroll support) to Meridian Components Ltd for a fixed fee of Rs 2.4 crore, payable in six equal monthly instalments.

## Clause 13.3 - Tax on transition fees
The Company shall withhold tax on transition service fees under Section 94Z of the Income Tax Act at the rate prevailing on the payment date. Any shortfall in withholding is a Pre-Closing Tax liability only for fees accrued before Closing.

## Clause 14 - Price adjustment
If the audited closing net working capital is below Rs 310 crore, the Purchase Price is reduced rupee for rupee, up to a maximum reduction of Rs 25 crore.

## Clause 15 - Valuation condition
The Buyer may require a refreshed report from a Registered Valuer if the Closing Date falls more than 90 days after the original valuation date.

## Clause 16 - Longstop date
Either party may terminate this Agreement if Closing has not occurred by 30 September 2026.
"""),
]

QUESTIONS = """\
# Questions to try after uploading (Tab 1 -> Ingest, then ask in the chat)

Sample_Tax_Circular_Section_94W.pdf  (practice area: Tax & Regulatory Services)
  1. What rate and threshold apply to Section 94W withholding on director sitting fees?
  2. Are reimbursements of travel costs for directors subject to Section 94W?
  3. What is the penalty under Section 271W?

Sample_GRC_Third_Party_Risk_SOP.docx  (practice area: Risk Advisory & GRC)
  4. How often must Tier 1 vendors be reviewed under SOP-TPR-05?
  5. Within how many hours must a vendor security incident reach the CISO?
  6. Which control maps to RBI outsourcing Para 11 and what penalty applies if it is missing?

Sample_Deal_SPA_Amendment_1_Meridian.docx  (practice area: Deal Advisory & Corporate Finance)
  7. What is the maximum purchase price reduction under the working-capital adjustment?
  8. What is the longstop date for Closing?
  -- Multi-hop (new file + existing corpus; use the Graph RAG engine, practice area 'All'): --
  9. Which provision governs withholding on the transition services fee and what rate applied to Meridian's earlier intra-group fees?
 10. Can the Buyer demand a new valuation if Closing slips, and what does the existing corpus say about how long a valuation report stays valid?

Tip: ask question 7-10 with Standard RAG and then Graph RAG to compare. Before uploading, ask question 1 and 7: the system should say they are not in the corpus.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, writer, area, title, body in SAMPLES:
        writer({"title": title}, body, OUT / fname)
        print("wrote", fname)
    (OUT / "SAMPLE_QUESTIONS.txt").write_text(QUESTIONS)
    print("wrote SAMPLE_QUESTIONS.txt")


if __name__ == "__main__":
    main()
