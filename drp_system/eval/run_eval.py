"""
Evaluate DRP system against cases_analysis_v2_clean_en.csv.

    python -m drp_system.rag.ingest_cases   # seed RAG first
    python -m drp_system.eval.run_eval --limit 3

Requires GEMINI_API_KEY in .env.
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from drp_system import config
from drp_system.agents.orchestrator import process_patient
from drp_system.eval.cases_loader import PRM_TO_PCNE, load_cases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(config.ROOT_DIR) / "eval" / "results"


def _score_case(prm_expected: str, report) -> dict:
    drp = report.drp_classification
    predicted_prefix = None
    if drp.category:
        predicted_prefix = drp.category.value.split(":")[0].strip()

    acceptable = PRM_TO_PCNE.get(prm_expected, set())
    category_match = predicted_prefix in acceptable if predicted_prefix else False

    return {
        "drp_present": drp.drp_present,
        "expected_prm": prm_expected,
        "predicted_category": str(drp.category) if drp.category else None,
        "predicted_pcne_prefix": predicted_prefix,
        "category_match": category_match,
        "drp_detected": drp.drp_present,
        "confidence": drp.confidence,
        "problem_description": drp.problem_description,
    }


async def _evaluate_cases(
    limit: int | None,
    case_ids: list[str] | None,
    concurrency: int,
) -> dict:
    cases = load_cases()
    if case_ids:
        id_set = set(case_ids)
        cases = [c for c in cases if c.case_id in id_set]
    if limit:
        cases = cases[:limit]

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def _run_one(case):
        async with semaphore:
            logger.info("Evaluating case %s (%s)", case.case_id, case.prm)
            try:
                report = await process_patient(
                    patient_id=f"CASE-{case.case_id}",
                    medications=case.medications,
                    disease=case.disease,
                )
                scores = _score_case(case.prm, report)
                return {
                    "case_id": case.case_id,
                    "disease": case.disease,
                    "prm_expected": case.prm,
                    "status": "ok",
                    "scores": scores,
                    "report": report.model_dump(mode="json"),
                }
            except Exception as exc:
                logger.exception("Case %s failed", case.case_id)
                return {
                    "case_id": case.case_id,
                    "prm_expected": case.prm,
                    "status": "error",
                    "error": str(exc),
                }

    tasks = [_run_one(c) for c in cases]
    results = await asyncio.gather(*tasks)

    ok = [r for r in results if r.get("status") == "ok"]
    category_hits = sum(1 for r in ok if r["scores"]["category_match"])
    drp_hits = sum(1 for r in ok if r["scores"]["drp_detected"])

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "successful": len(ok),
        "errors": len(results) - len(ok),
        "drp_detection_rate": drp_hits / len(ok) if ok else 0.0,
        "prm_category_match_rate": category_hits / len(ok) if ok else 0.0,
        "cases": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DRP system on clinical cases")
    parser.add_argument("--limit", type=int, help="Max cases to run")
    parser.add_argument("--case-id", action="append", dest="case_ids", help="Specific case ID(s)")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel patients (keep low for 10 RPM Gemini limit)",
    )
    parser.add_argument(
        "--output",
        help="Results JSON path (default: eval/results/run_<timestamp>.json)",
    )
    args = parser.parse_args()

    summary = asyncio.run(
        _evaluate_cases(args.limit, args.case_ids, args.concurrency)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.output
    if not out:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = str(RESULTS_DIR / f"run_{ts}.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {out}")
    print(f"Successful: {summary['successful']}/{summary['total']}")
    print(f"DRP detection rate: {summary['drp_detection_rate']:.1%}")
    print(f"PRM→PCNE category match: {summary['prm_category_match_rate']:.1%}")


if __name__ == "__main__":
    main()
