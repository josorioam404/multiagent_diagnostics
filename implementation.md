# DRP Multiagent System — Implementation Guide

## Overview

This document describes the full implementation of a Drug-Related Problem (DRP) classification system using a multiagent architecture powered by the Gemini 2.5 Flash free API, a local RAG pipeline (ChromaDB + sentence-transformers), and the openFDA REST API for live drug label data.

**Classification framework:** PCNE v9.1 (Pharmaceutical Care Network Europe) with Cipolle categories as supplementary reference.

**Hard constraints:**
- All components free (no paid APIs)
- Gemini 2.5 Flash: 10 RPM, 250 RPD, 250,000 TPM
- Target: 2–3 Gemini calls per patient (not 4) — decomposition done in Python

---

## Project Structure

```
drp_system/
├── agents/
│   ├── orchestrator.py          # Synthesizes agent outputs → DRP report
│   ├── medication_agent.py      # Drug research: interactions, contraindications
│   └── disease_agent.py         # Disease research: guidelines, drug targets
├── rag/
│   ├── ingest.py                # Chunk PDFs/CSVs and embed into ChromaDB
│   ├── retriever.py             # Semantic search interface over ChromaDB
│   └── openfda.py               # Live drug label lookup (no key required)
├── llm/
│   └── gemini_client.py         # Shared Gemini client with tenacity retry
├── models/
│   └── drp_schema.py            # Pydantic models for DRP output
├── data/
│   └── knowledge_base/          # Place source PDFs and CSVs here
│       ├── who_formulary.pdf
│       ├── cipolle_excerpts.pdf  # Manual excerpts from Cipolle 4th ed.
│       └── drugbank_open.csv    # DrugBank open vocabulary (free tier)
├── config.py                    # API key, model name, paths
├── main.py                      # Entry point
└── requirements.txt
```

---

## Dependencies

```txt
# requirements.txt
google-generativeai>=0.8.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
pymupdf>=1.24.0
pydantic>=2.0.0
tenacity>=8.3.0
httpx>=0.27.0
python-dotenv>=1.0.0
```

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]   # From aistudio.google.com — free
GEMINI_MODEL   = "gemini-2.5-flash"

CHROMA_PATH    = "./data/chroma_db"              # Local vector store directory
EMBED_MODEL    = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
                 # Biomedical embedding model — better domain fit than MiniLM
                 # Fallback: "sentence-transformers/all-MiniLM-L6-v2"

KNOWLEDGE_BASE_DIR = "./data/knowledge_base"

RAG_TOP_K      = 5    # Chunks retrieved per query
CHUNK_SIZE     = 600  # Characters per chunk
CHUNK_OVERLAP  = 80
```

---

## Step 1 — DRP Output Schema

Define the output structure before writing any agent. This schema drives every prompt.

```python
# models/drp_schema.py
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DRPCategory(str, Enum):
    """PCNE v9.1 Problem Classification"""
    P1 = "P1: Adverse drug reaction"
    P2 = "P2: Drug choice problem"
    P3 = "P3: Dosing problem"
    P4 = "P4: Drug use problem"
    P5 = "P5: Drug interactions"
    P6 = "P6: Other drug-related problem"
    P7 = "P7: Untreated indication"


class DRPCause(str, Enum):
    C1 = "C1: Drug selection"
    C2 = "C2: Drug form selection"
    C3 = "C3: Dose selection"
    C4 = "C4: Treatment duration"
    C5 = "C5: Drug use process"
    C6 = "C6: Logistics"
    C7 = "C7: Patient related"
    C8 = "C8: Other cause"


class Severity(str, Enum):
    MINOR    = "minor"
    MODERATE = "moderate"
    MAJOR    = "major"


class MedicationFinding(BaseModel):
    medication: str
    finding: str                   # What was found (interaction, contraindication, etc.)
    evidence_source: str           # Which knowledge base chunk supported this
    severity: Severity


class DRPClassification(BaseModel):
    drp_present: bool
    category: Optional[DRPCategory] = None
    cause: Optional[DRPCause] = None
    severity: Optional[Severity] = None
    problem_description: str
    implicated_medications: list[str]
    disease_context: str           # How the disease modifies the risk
    recommended_intervention: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str          # Key RAG chunks that drove the conclusion
    disclaimer: str = (
        "This output is for decision-support only and must be reviewed "
        "by a licensed pharmacist or physician before clinical use."
    )


class PatientReport(BaseModel):
    patient_id: str
    medications: list[str]
    disease: str
    medication_findings: list[MedicationFinding]
    drp_classification: DRPClassification
```

---

## Step 2 — Gemini Client with Rate-Limit Handling

```python
# llm/gemini_client.py
import google.generativeai as genai
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import json
import config

genai.configure(api_key=config.GEMINI_API_KEY)

_model = genai.GenerativeModel(
    model_name=config.GEMINI_MODEL,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json",  # Forces structured JSON output
        temperature=0.2,                        # Low temp for clinical consistency
        max_output_tokens=2048,
    ),
)


@retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_exponential(multiplier=1, min=5, max=90),
    stop=stop_after_attempt(4),
)
async def call_gemini(prompt: str) -> dict:
    """
    Single async Gemini call with automatic retry on 429 / 503.
    Always returns parsed JSON dict.
    """
    response = await _model.generate_content_async(prompt)
    text = response.text.strip()

    # Strip markdown fences if Gemini wraps JSON despite mime type
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text)
```

---

## Step 3 — RAG Pipeline

### 3a — Ingestion

```python
# rag/ingest.py
"""
Run once to build the vector store:
    python -m rag.ingest
"""
import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 50]  # Drop tiny fragments


def ingest_pdf(path: str) -> list[str]:
    doc = fitz.open(path)
    full_text = "\n".join(page.get_text() for page in doc)
    return chunk_text(full_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


def ingest_csv(path: str) -> list[str]:
    """For DrugBank open CSV: combine relevant columns into text chunks."""
    import csv
    chunks = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Adjust column names to match your DrugBank CSV headers
            name        = row.get("Name", "")
            description = row.get("Description", "")
            indication  = row.get("Indication", "")
            interactions = row.get("Drug Interactions", "")
            text = (
                f"Drug: {name}\n"
                f"Description: {description}\n"
                f"Indication: {indication}\n"
                f"Interactions: {interactions}"
            )
            chunks.extend(chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
    return chunks


def build_vector_store():
    embedder = SentenceTransformer(config.EMBED_MODEL)
    client   = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection("drp_knowledge")

    all_chunks: list[str] = []
    kb_dir = config.KNOWLEDGE_BASE_DIR

    for fname in os.listdir(kb_dir):
        fpath = os.path.join(kb_dir, fname)
        print(f"Ingesting: {fname}")
        if fname.endswith(".pdf"):
            all_chunks.extend(ingest_pdf(fpath))
        elif fname.endswith(".csv"):
            all_chunks.extend(ingest_csv(fpath))

    print(f"Total chunks: {len(all_chunks)}")

    # Embed in batches of 64 to avoid memory spikes
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings = embedder.encode(batch).tolist()
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        collection.add(documents=batch, embeddings=embeddings, ids=ids)
        print(f"  Stored {i + len(batch)}/{len(all_chunks)} chunks")

    print("Vector store built successfully.")


if __name__ == "__main__":
    build_vector_store()
```

### 3b — Retriever

```python
# rag/retriever.py
import chromadb
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import config


@lru_cache(maxsize=1)
def _get_resources():
    """Lazy-load embedder and ChromaDB — cached as singletons."""
    embedder   = SentenceTransformer(config.EMBED_MODEL)
    client     = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_collection("drp_knowledge")
    return embedder, collection


def retrieve(query: str, top_k: int = None) -> list[str]:
    """Return top_k relevant text chunks for a given query."""
    k = top_k or config.RAG_TOP_K
    embedder, collection = _get_resources()

    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    return results["documents"][0]  # list[str]
```

### 3c — openFDA Live Lookup

```python
# rag/openfda.py
"""
Live drug label lookup from openFDA — no API key required.
Used to supplement RAG with current FDA-approved label data.
"""
import httpx
from tenacity import retry, wait_fixed, stop_after_attempt

BASE_URL = "https://api.fda.gov/drug/label.json"


@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
async def get_drug_label(drug_name: str) -> dict:
    """
    Fetch FDA drug label for a medication.
    Returns dict with keys: warnings, contraindications, drug_interactions, indications_and_usage.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            BASE_URL,
            params={"search": f'openfda.brand_name:"{drug_name}"', "limit": 1},
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("results"):
        # Try generic name as fallback
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                BASE_URL,
                params={"search": f'openfda.generic_name:"{drug_name}"', "limit": 1},
            )
            resp.raise_for_status()
            data = resp.json()

    if not data.get("results"):
        return {"error": f"No FDA label found for '{drug_name}'"}

    label = data["results"][0]
    return {
        "warnings":             label.get("warnings", ["Not found"]),
        "contraindications":    label.get("contraindications", ["Not found"]),
        "drug_interactions":    label.get("drug_interactions", ["Not found"]),
        "indications_and_usage": label.get("indications_and_usage", ["Not found"]),
    }
```

---

## Step 4 — Agents

### Medication Agent

```python
# agents/medication_agent.py
import asyncio
from rag.retriever import retrieve
from rag.openfda import get_drug_label
from llm.gemini_client import call_gemini


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


async def run_medication_agent(
    medications: list[str],
    disease: str,
) -> dict:
    """
    Gemini Call #1 (of 2 total).
    RAG queries + openFDA are local/free — only the Gemini call counts against quota.
    """
    # Local RAG — free, no rate limits
    rag_query   = f"drug interactions contraindications {' '.join(medications)} {disease}"
    rag_chunks  = retrieve(rag_query)
    rag_context = "\n---\n".join(rag_chunks)

    # Live FDA labels — free REST API
    fda_results = await asyncio.gather(
        *[get_drug_label(med) for med in medications],
        return_exceptions=True,
    )
    fda_data = {
        med: result if not isinstance(result, Exception) else {"error": str(result)}
        for med, result in zip(medications, fda_results)
    }

    prompt = MEDICATION_AGENT_PROMPT.format(
        medications="\n".join(f"- {m}" for m in medications),
        disease=disease,
        rag_context=rag_context,
        fda_data=str(fda_data),
    )

    return await call_gemini(prompt)  # 1 Gemini call
```

### Disease Agent

```python
# agents/disease_agent.py
from rag.retriever import retrieve
from llm.gemini_client import call_gemini


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
  "overall_disease_assessment": "string — paragraph summary"
}}
"""


async def run_disease_agent(
    disease: str,
    medications: list[str],
) -> dict:
    """
    Gemini Call #2 (of 2 total).
    """
    rag_query   = f"treatment guidelines {disease} pharmacotherapy drug selection"
    rag_chunks  = retrieve(rag_query)
    rag_context = "\n---\n".join(rag_chunks)

    prompt = DISEASE_AGENT_PROMPT.format(
        disease=disease,
        medications="\n".join(f"- {m}" for m in medications),
        rag_context=rag_context,
    )

    return await call_gemini(prompt)  # 1 Gemini call
```

---

## Step 5 — Orchestrator

The orchestrator dispatches both agents in parallel (`asyncio.gather`), then performs the final synthesis. Decomposition is pure Python — no LLM call needed.

```python
# agents/orchestrator.py
import asyncio
from agents.medication_agent import run_medication_agent
from agents.disease_agent import run_disease_agent
from llm.gemini_client import call_gemini
from models.drp_schema import PatientReport, DRPClassification, MedicationFinding, Severity
import json


SYNTHESIS_PROMPT = """
You are a senior clinical pharmacist performing a Drug-Related Problem (DRP) classification using the PCNE v9.1 framework.

PATIENT INFORMATION:
- Medications: {medications}
- Diagnosed disease: {disease}

MEDICATION AGENT REPORT:
{medication_report}

DISEASE AGENT REPORT:
{disease_report}

Based on both reports, determine the primary DRP (if any) and classify it.

Return ONLY valid JSON matching this schema exactly:
{{
  "drp_present": true|false,
  "category": "P1: Adverse drug reaction|P2: Drug choice problem|P3: Dosing problem|P4: Drug use problem|P5: Drug interactions|P6: Other drug-related problem|P7: Untreated indication|null",
  "cause": "C1: Drug selection|C2: Drug form selection|C3: Dose selection|C4: Treatment duration|C5: Drug use process|C6: Logistics|C7: Patient related|C8: Other cause|null",
  "severity": "minor|moderate|major|null",
  "problem_description": "string — clear clinical description of the DRP",
  "implicated_medications": ["string"],
  "disease_context": "string — how the disease modifies the risk or problem",
  "recommended_intervention": "string — specific, actionable recommendation",
  "confidence": 0.0 to 1.0,
  "evidence_summary": "string — key evidence from both agent reports"
}}

If no DRP is present, set drp_present to false and all classification fields to null.
"""


async def process_patient(
    patient_id: str,
    medications: list[str],
    disease: str,
) -> PatientReport:
    """
    Main orchestration function.
    Total Gemini calls: 2 (agents in parallel) + 1 (synthesis) = 3 max.
    Decomposition is Python — free, instant.
    """

    # Step 1: Dispatch both agents in parallel — 2 concurrent Gemini calls
    medication_result, disease_result = await asyncio.gather(
        run_medication_agent(medications, disease),
        run_disease_agent(disease, medications),
    )

    # Step 2: Synthesize — 1 Gemini call
    synthesis_prompt = SYNTHESIS_PROMPT.format(
        medications=", ".join(medications),
        disease=disease,
        medication_report=json.dumps(medication_result, indent=2),
        disease_report=json.dumps(disease_result, indent=2),
    )
    synthesis_result = await call_gemini(synthesis_prompt)

    # Step 3: Build structured Pydantic output — pure Python
    med_findings = [
        MedicationFinding(
            medication=f["medication"],
            finding=f["finding"],
            evidence_source=f["evidence_source"],
            severity=Severity(f["severity"]),
        )
        for f in medication_result.get("findings", [])
    ]

    drp = DRPClassification(
        drp_present=synthesis_result["drp_present"],
        category=synthesis_result.get("category"),
        cause=synthesis_result.get("cause"),
        severity=synthesis_result.get("severity"),
        problem_description=synthesis_result["problem_description"],
        implicated_medications=synthesis_result["implicated_medications"],
        disease_context=synthesis_result["disease_context"],
        recommended_intervention=synthesis_result["recommended_intervention"],
        confidence=synthesis_result["confidence"],
        evidence_summary=synthesis_result["evidence_summary"],
    )

    return PatientReport(
        patient_id=patient_id,
        medications=medications,
        disease=disease,
        medication_findings=med_findings,
        drp_classification=drp,
    )
```

---

## Step 6 — Entry Point

```python
# main.py
import asyncio
import json
from agents.orchestrator import process_patient


async def main():
    # Example patient
    report = await process_patient(
        patient_id="PT-001",
        medications=[
            "Metformin 850mg",
            "Lisinopril 10mg",
            "Ibuprofen 400mg",
            "Simvastatin 20mg",
        ],
        disease="Type 2 Diabetes Mellitus with early-stage chronic kidney disease (CKD stage 3)",
    )

    print(json.dumps(report.model_dump(), indent=2))

    # Also save to file
    with open("output_report.json", "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    print("\n--- DRP SUMMARY ---")
    drp = report.drp_classification
    print(f"DRP Present:     {drp.drp_present}")
    print(f"Category:        {drp.category}")
    print(f"Cause:           {drp.cause}")
    print(f"Severity:        {drp.severity}")
    print(f"Problem:         {drp.problem_description}")
    print(f"Intervention:    {drp.recommended_intervention}")
    print(f"Confidence:      {drp.confidence:.0%}")
    print(f"\nDisclaimer: {drp.disclaimer}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 7 — Build Order

Follow this sequence to avoid integration surprises:

```
1. pip install -r requirements.txt
2. Get Gemini API key → https://aistudio.google.com/app/apikey
3. Create .env with GEMINI_API_KEY=your_key_here
4. Place knowledge base files in data/knowledge_base/
5. python -m rag.ingest          ← builds ChromaDB (run once)
6. python main.py                ← test with example patient
```

To verify RAG quality before running the full pipeline:

```python
# Quick RAG test — run this interactively
from rag.retriever import retrieve
results = retrieve("metformin contraindication renal failure")
for i, chunk in enumerate(results):
    print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}")
```

---

## Knowledge Base Sources (All Free)

| Source | Format | Access | Notes |
|---|---|---|---|
| WHO Model Formulary 2008 | PDF | [who.int](https://www.who.int/publications/i/item/978924154765) | Open access |
| DrugBank Open Vocabulary | CSV | [drugbank.com/releases/latest](https://go.drugbank.com/releases/latest#open-data) | Free tier registration |
| openFDA Drug Labels | REST API | No key required | Live, always current |
| Cipolle 4th ed. excerpts | PDF | Manual — your own copy | Copyright; excerpt only relevant chapters |
| PCNE v9.1 classification | PDF | [pcne.org](https://www.pcne.org/working-groups/2/drug-related-problem-classification) | Free download |

> **Note on Cipolle:** The full textbook cannot be redistributed. Extract and embed only the DRP classification tables and framework chapters (typically ~30 pages) that you have legal access to.

---

## Rate Limit Strategy

| Limit | Value | Implication |
|---|---|---|
| RPM | 10 | Two parallel agent calls = 2 RPM consumed together |
| RPD | 250 | 3 calls/patient → ~83 patients/day max |
| TPM | 250,000 | With RAG context ~800 tokens/call → well within limits |

**Mitigation built in:**
- `tenacity` retries with exponential backoff (5s → 90s) on 429 and 503 errors
- `asyncio.gather` for parallel agents reduces wall-clock time without exceeding RPM (two simultaneous calls still counts as 2 RPM, not serialized over 2 minutes)
- `response_mime_type="application/json"` eliminates format-parsing retries
- Low temperature (0.2) reduces output variance and re-run needs

---

## Expected Output Shape

```json
{
  "patient_id": "PT-001",
  "medications": ["Metformin 850mg", "Lisinopril 10mg", "Ibuprofen 400mg", "Simvastatin 20mg"],
  "disease": "Type 2 Diabetes Mellitus with early-stage CKD stage 3",
  "medication_findings": [
    {
      "medication": "Ibuprofen 400mg",
      "finding": "NSAIDs are contraindicated in CKD stage 3+ due to nephrotoxicity risk and may accelerate renal function decline. Also reduces efficacy of ACE inhibitors like Lisinopril.",
      "evidence_source": "WHO Formulary: NSAIDs; FDA label: ibuprofen contraindications",
      "severity": "major"
    }
  ],
  "drp_classification": {
    "drp_present": true,
    "category": "P2: Drug choice problem",
    "cause": "C1: Drug selection",
    "severity": "major",
    "problem_description": "Ibuprofen is contraindicated in CKD stage 3 and antagonizes Lisinopril's renoprotective effect in diabetic nephropathy.",
    "implicated_medications": ["Ibuprofen 400mg", "Lisinopril 10mg"],
    "disease_context": "CKD stage 3 significantly increases nephrotoxicity risk from NSAIDs; diabetic nephropathy management requires ACE inhibitor efficacy preservation.",
    "recommended_intervention": "Discontinue Ibuprofen. Switch to Paracetamol (Acetaminophen) ≤3g/day for pain management. Monitor eGFR at next visit.",
    "confidence": 0.94,
    "evidence_summary": "FDA label contraindication confirmed; WHO Formulary corroborates NSAID avoidance in renal impairment; Lisinopril-NSAID interaction documented in both RAG sources.",
    "disclaimer": "This output is for decision-support only and must be reviewed by a licensed pharmacist or physician before clinical use."
  }
}
```

---

## Extending the System

**Add more agents:**  
Create `agents/interaction_checker_agent.py` following the same pattern — one RAG query, one Gemini call, structured JSON output.

**Support multiple DRPs:**  
Change `DRPClassification` to `list[DRPClassification]` in `PatientReport` and update the synthesis prompt to return an array.

**Add a CLI:**  
```python
python main.py --patient-id PT-002 \
               --medications "Warfarin 5mg" "Aspirin 100mg" \
               --disease "Atrial fibrillation post-MI"
```

**Batch processing:**  
Use `asyncio.Semaphore(2)` to cap concurrent patients and respect the 10 RPM limit across a batch run.

**Swap to Flash-Lite for high volume:**  
Change `GEMINI_MODEL = "gemini-2.5-flash-lite"` in `config.py` — same code, 4× daily quota, lower reasoning depth.
