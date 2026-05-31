"""Entry point: run DRP analysis for one patient."""
import argparse
import asyncio
import json
import logging
import sys

from drp_system.agents.orchestrator import process_patient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


async def _run(args: argparse.Namespace) -> None:
    medications = args.medications
    if not medications:
        medications = [
            "Metformin 850mg",
            "Lisinopril 10mg",
            "Ibuprofen 400mg",
            "Simvastatin 20mg",
        ]
    disease = args.disease or (
        "Type 2 Diabetes Mellitus with early-stage chronic kidney disease (CKD stage 3)"
    )

    report = await process_patient(
        patient_id=args.patient_id,
        medications=medications,
        disease=disease,
    )

    payload = report.model_dump(mode="json")
    print(json.dumps(payload, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.output}")

    drp = report.drp_classification
    print("\n--- DRP SUMMARY ---")
    print(f"DRP Present:     {drp.drp_present}")
    print(f"Category:        {drp.category}")
    print(f"Cause:           {drp.cause}")
    print(f"Severity:        {drp.severity}")
    print(f"Problem:         {drp.problem_description}")
    print(f"Intervention:    {drp.recommended_intervention}")
    print(f"Confidence:      {drp.confidence:.0%}")
    print(f"\nDisclaimer: {drp.disclaimer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRP multiagent classification")
    parser.add_argument("--patient-id", default="PT-001")
    parser.add_argument(
        "--medications",
        nargs="+",
        help="Medication list (space-separated)",
    )
    parser.add_argument("--disease", help="Diagnosed disease / health situation")
    parser.add_argument(
        "--output",
        default="output_report.json",
        help="JSON output path",
    )
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
