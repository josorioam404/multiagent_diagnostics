import asyncio
import json
import logging

from drp_system.agents.disease_agent import run_disease_agent
from drp_system.agents.medication_agent import run_medication_agent
from drp_system import config
from drp_system.llm.gemini_client import call_gemini, truncate_context
from drp_system.models.drp_schema import (
    DRPCause,
    DRPCategory,
    DRPClassification,
    MedicationFinding,
    PatientReport,
    Severity,
)

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """
You are a senior clinical pharmacist performing a Drug-Related Problem (DRP) classification using the PCNE v9.1 framework.

PATIENT INFORMATION:
- Medications: {medications}
- Diagnosed disease: {disease}

CLINICAL HISTORY (patient-specific context — labs, vitals, adherence, course):
{clinical_history}

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
  "confidence": 0.0,
  "evidence_summary": "string — key evidence from both agent reports"
}}

If no DRP is present, set drp_present to false and classification fields to null.

Use clinical history together with agent reports. Agent reports did not see clinical history.
"""


def _format_clinical_history(clinical_history: str | None) -> str:
    if not clinical_history or not clinical_history.strip():
        return "(not provided)"
    return truncate_context(
        clinical_history.strip(),
        config.CLINICAL_HISTORY_MAX_CHARS,
    )


def _parse_severity(value: str | None) -> Severity | None:
    if not value or value == "null":
        return None
    try:
        return Severity(str(value).lower())
    except ValueError:
        return Severity.MODERATE


def _parse_category(value: str | None) -> DRPCategory | None:
    if not value or value == "null":
        return None
    for cat in DRPCategory:
        if value.strip().startswith(cat.value[:2]) or cat.value in value:
            return cat
    return None


def _parse_cause(value: str | None) -> DRPCause | None:
    if not value or value == "null":
        return None
    for cause in DRPCause:
        if value.strip().startswith(cause.value[:2]) or cause.value in value:
            return cause
    return None


def _build_medication_findings(medication_result: dict) -> list[MedicationFinding]:
    findings: list[MedicationFinding] = []
    for f in medication_result.get("findings", []):
        try:
            severity = Severity(str(f.get("severity", "moderate")).lower())
        except ValueError:
            severity = Severity.MODERATE
        findings.append(
            MedicationFinding(
                medication=f.get("medication", "unknown"),
                finding=f.get("finding", ""),
                evidence_source=f.get("evidence_source", ""),
                severity=severity,
            )
        )
    return findings


async def process_patient(
    patient_id: str,
    medications: list[str],
    disease: str,
    clinical_history: str | None = None,
) -> PatientReport:
    """
    Orchestrate DRP analysis: 2 parallel agent calls + 1 synthesis (3 Gemini calls).

    clinical_history is passed only to the synthesis step, not to specialist agents.
    """
    logger.info("Processing patient %s (3 Gemini calls)", patient_id)

    medication_result, disease_result = await asyncio.gather(
        run_medication_agent(medications, disease),
        run_disease_agent(disease, medications),
    )

    history_text = _format_clinical_history(clinical_history)
    if clinical_history and clinical_history.strip():
        logger.info("Clinical history included in synthesis (%d chars)", len(history_text))

    synthesis_prompt = SYNTHESIS_PROMPT.format(
        medications=", ".join(medications),
        disease=disease,
        clinical_history=history_text,
        medication_report=json.dumps(medication_result, indent=2),
        disease_report=json.dumps(disease_result, indent=2),
    )
    synthesis_result = await call_gemini(synthesis_prompt)

    drp = DRPClassification(
        drp_present=bool(synthesis_result.get("drp_present", False)),
        category=_parse_category(synthesis_result.get("category")),
        cause=_parse_cause(synthesis_result.get("cause")),
        severity=_parse_severity(synthesis_result.get("severity")),
        problem_description=synthesis_result.get("problem_description", ""),
        implicated_medications=synthesis_result.get("implicated_medications") or [],
        disease_context=synthesis_result.get("disease_context", ""),
        recommended_intervention=synthesis_result.get("recommended_intervention", ""),
        confidence=float(synthesis_result.get("confidence", 0.0)),
        evidence_summary=synthesis_result.get("evidence_summary", ""),
    )

    return PatientReport(
        patient_id=patient_id,
        medications=medications,
        disease=disease,
        clinical_history=clinical_history or "",
        medication_findings=_build_medication_findings(medication_result),
        drp_classification=drp,
    )
