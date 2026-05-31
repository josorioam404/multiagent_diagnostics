# DRP Multiagent System — Implementation Plan

Track progress against milestones. Source spec: [implementation.md](implementation.md).

**Test corpus:** [cases_analysis_v2_clean_en.csv](cases_analysis_v2_clean_en.csv) (49 clinical cases; ground-truth `PRM`: Safety, Adherence, Efficacy, Indication).

## Milestones

| # | Milestone | Status | Exit criteria |
|---|-----------|--------|---------------|
| M0 | Project bootstrap | Done | Installable package, `.env.example`, `.gitignore` |
| M1 | Data contracts | Done | `drp_schema.py`, `config.py` |
| M2 | LLM layer | Done | `call_gemini` with retry + JSON parse |
| M3 | RAG pipeline | Done | ingest, retriever, openFDA |
| M4 | Specialist agents | Done | medication + disease agents |
| M5 | Orchestration | Done | 3 Gemini calls → `PatientReport` |
| M6 | Knowledge base & QA | In progress | Full PDF ingest + RAG regression |
| M7 | Hardening | Planned | CLI, batch semaphore, multi-DRP |

## Architecture

```
Patient (meds + disease)
  ├─► Medication agent ── RAG + openFDA ──► Gemini #1 ─┐
  ├─► Disease agent    ── RAG           ──► Gemini #2 ─┼─► Orchestrator ──► Gemini #3 ──► PatientReport
```

**Quota:** 3 Gemini calls/patient (2 parallel + 1 synthesis).

## Task backlog

### M6 — Knowledge base & validation

- [ ] Add WHO formulary, PCNE PDF, DrugBank CSV under `data/knowledge_base/`
- [ ] `python -m drp_system.rag.ingest`
- [ ] RAG smoke queries (metformin+CKD, NSAID+renal)
- [ ] Run `python -m drp_system.eval.run_eval` on full CSV; review metrics

### M7 — Hardening

- [ ] CLI flags on `main.py`
- [ ] `asyncio.Semaphore(2)` for batch eval (10 RPM)
- [ ] Optional `gemini-2.5-flash-lite` in config

## PRM → PCNE mapping (evaluation)

| Ground truth `PRM` | Expected PCNE category (primary) |
|--------------------|----------------------------------|
| Safety | P1, P2, P5, or P6 |
| Adherence | P4 |
| Efficacy | P2 or P3 |
| Indication | P7 |

## Acceptance checklist (MVP)

- [x] Pydantic `PatientReport` validates end-to-end
- [x] 3 Gemini calls per patient (logged)
- [x] Eval runner loads CSV and writes `eval/results/`
- [ ] RAG ingest on full knowledge base
- [ ] PRM alignment ≥ target on case suite (tune after KB)

## Risks

| Risk | Mitigation |
|------|------------|
| 10 RPM / 250 RPD | Semaphore in eval; exponential backoff |
| Empty Chroma | `ingest_cases` seeds from CSV; full `ingest` for PDFs |
| openFDA name mismatch | `normalize_drug_name()` in medication agent |
| Enum string mismatch | `parse_pcne_category()` / `parse_pcne_cause()` in orchestrator |

## Commit convention

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): Imperative subject` with body explaining *why* when non-obvious.
