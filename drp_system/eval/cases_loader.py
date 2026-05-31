"""Load clinical test cases from cases_analysis_v2_clean_en.csv."""
import csv
from dataclasses import dataclass

from drp_system import config


@dataclass
class ClinicalCase:
    case_id: str
    medications_raw: str
    disease: str
    folio: str
    prm: str  # Safety | Adherence | Efficacy | Indication
    justification: str
    clinical_evidence: str

    @property
    def medications(self) -> list[str]:
        """Return medication tokens for the pipeline."""
        raw = self.medications_raw.strip()
        if raw.lower() in ("none", "none (patient without treatment due to lack of adherence)"):
            return []
        return [raw]


# PRM (case suite labels) → acceptable PCNE P-categories
PRM_TO_PCNE: dict[str, set[str]] = {
    "Safety": {"P1", "P2", "P5", "P6"},
    "Adherence": {"P4"},
    "Efficacy": {"P2", "P3"},
    "Indication": {"P7"},
}


def load_cases(path: str | None = None) -> list[ClinicalCase]:
    csv_path = path or config.CASES_CSV_PATH
    cases: list[ClinicalCase] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            case_id = (row.get("ID") or "").strip()
            if not case_id:
                continue
            cases.append(
                ClinicalCase(
                    case_id=case_id,
                    medications_raw=row.get("LM", ""),
                    disease=row.get("HS", ""),
                    folio=row.get("PG", ""),
                    prm=row.get("PRM", "").strip(),
                    justification=row.get("Jus", ""),
                    clinical_evidence=row.get("TE", ""),
                )
            )
    return cases


def pcne_prefix(category_value: str | None) -> str | None:
    if not category_value:
        return None
    return category_value.split(":")[0].strip()
