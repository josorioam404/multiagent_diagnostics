import asyncio
import json

from drp_system.llm.gemini_client import call_gemini, normalize_drug_name, truncate_context
from drp_system.rag.openfda import get_drug_label
from drp_system.rag.retriever import retrieve

MEDICATION_AGENT_PROMPT = """
You are a clinical pharmacist AI analyzing drug-related problems.

PATIENT MEDICATIONS:
{medications}

DIAGNOSED DISEASE:
{disease}

KNOWLEDGE BASE CONTEXT (RAG):
{rag_context}

FDA LABEL DATA:
{fda_data}

Analyze each medication for:
1. Contraindications given the diagnosed disease
2. Drug-drug interactions among the listed medications
3. Dosing appropriateness concerns
4. Adverse effects relevant to this patient's condition

Return ONLY valid JSON matching this schema:
{{
  "findings": [
    {{
      "medication": "string",
      "finding": "string — specific clinical concern",
      "evidence_source": "string — which context supported this",
      "severity": "minor|moderate|major"
    }}
  ],
  "overall_drug_assessment": "string — paragraph summary"
}}

Be specific and cite evidence. If no DRP found for a medication, still include it with finding = "No significant DRP identified".
"""


def _parse_medications(medications: list[str]) -> list[str]:
    """Expand comma-separated med strings into individual entries."""
    items: list[str] = []
    for med in medications:
        for part in med.replace(";", ",").split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


async def run_medication_agent(medications: list[str], disease: str) -> dict:
    """Gemini call #1 — medication analysis with RAG + openFDA."""
    med_list = _parse_medications(medications)
    rag_query = f"drug interactions contraindications {' '.join(med_list)} {disease}"
    rag_chunks = retrieve(rag_query)
    rag_context = truncate_context(
        "\n---\n".join(rag_chunks) if rag_chunks else "(no RAG context — run ingest)"
    )

    unique_names = list(dict.fromkeys(normalize_drug_name(m) for m in med_list))
    fda_results = await asyncio.gather(
        *[get_drug_label(name) for name in unique_names],
        return_exceptions=True,
    )
    fda_data = {
        name: (result if not isinstance(result, Exception) else {"error": str(result)})
        for name, result in zip(unique_names, fda_results)
    }

    prompt = MEDICATION_AGENT_PROMPT.format(
        medications="\n".join(f"- {m}" for m in med_list),
        disease=disease,
        rag_context=rag_context,
        fda_data=json.dumps(fda_data, indent=2)[:12000],
    )
    return await call_gemini(prompt)
