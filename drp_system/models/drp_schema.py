from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DRPCategory(str, Enum):
    """PCNE v9.1 Problem Classification."""

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
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"


class MedicationFinding(BaseModel):
    medication: str
    finding: str
    evidence_source: str
    severity: Severity


class DRPClassification(BaseModel):
    drp_present: bool
    category: Optional[DRPCategory] = None
    cause: Optional[DRPCause] = None
    severity: Optional[Severity] = None
    problem_description: str = ""
    implicated_medications: list[str] = Field(default_factory=list)
    disease_context: str = ""
    recommended_intervention: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_summary: str = ""
    disclaimer: str = (
        "This output is for decision-support only and must be reviewed "
        "by a licensed pharmacist or physician before clinical use."
    )


class PatientReport(BaseModel):
    patient_id: str
    medications: list[str]
    disease: str
    clinical_history: str = ""
    medication_findings: list[MedicationFinding]
    drp_classification: DRPClassification
