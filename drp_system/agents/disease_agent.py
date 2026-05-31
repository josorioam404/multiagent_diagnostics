from drp_system.llm.gemini_client import call_gemini, truncate_context
from drp_system.rag.retriever import retrieve

DISEASE_AGENT_PROMPT = """
You are a clinical pharmacist AI specializing in disease-drug relationships.

DIAGNOSED DISEASE:
{disease}

PATIENT MEDICATIONS:
{medications}

KNOWLEDGE BASE CONTEXT (RAG):
{rag_context}

Analyze the disease from a pharmacological perspective:
1. Standard drug classes recommended for this disease (are they present?)
2. Drug classes that are contraindicated in this disease
3. Disease-related pharmacokinetic or pharmacodynamic changes that affect drug response
4. Untreated indications — conditions implied by this disease that may need additional therapy
5. Therapeutic goals and whether the current medication list aligns with them

Return ONLY valid JSON matching this schema:
{{
  "disease_profile": {{
    "recommended_drug_classes": ["string"],
    "contraindicated_drug_classes": ["string"],
    "pk_pd_considerations": "string",
    "untreated_indications": ["string"],
    "alignment_assessment": "string — does the medication list align with guidelines?"
  }},
  "disease_drug_conflicts": [
    {{
      "medication": "string",
      "conflict_type": "string",
      "explanation": "string",
      "severity": "minor|moderate|major"
    }}
  ],
  "overall_disease_assessment": "string — brief summary (max 3 sentences)"
}}

Keep all string fields concise so the JSON response fits in one message.
"""


async def run_disease_agent(disease: str, medications: list[str]) -> dict:
    """Gemini call #2 — disease-drug relationship analysis."""
    rag_query = f"treatment guidelines {disease} pharmacotherapy drug selection"
    rag_chunks = retrieve(rag_query)
    rag_context = truncate_context(
        "\n---\n".join(rag_chunks) if rag_chunks else "(no RAG context — run ingest)"
    )

    med_lines = []
    for med in medications:
        for part in med.replace(";", ",").split(","):
            part = part.strip()
            if part:
                med_lines.append(f"- {part}")

    prompt = DISEASE_AGENT_PROMPT.format(
        disease=disease,
        medications="\n".join(med_lines),
        rag_context=rag_context,
    )
    return await call_gemini(prompt)
