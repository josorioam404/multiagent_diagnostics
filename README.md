# IA Médicos — DRP Multiagent System

Drug-Related Problem (DRP) classification using **PCNE v9.1**, a multiagent pipeline (Gemini 2.5 Flash), local **RAG** (ChromaDB), and **openFDA** labels.

## Quick start

Uses [pyenv](https://github.com/pyenv/pyenv). The repo pins **Python 3.12.11** via `.python-version`.

```bash
cd /path/to/iamedicos
pyenv install -s 3.12.11   # skip if already installed
pyenv local 3.12.11        # optional if .python-version is present

python -m pip install -r requirements.txt
cp .env.example .env       # add GEMINI_API_KEY

# Seed vector store from test cases (or add PDFs under data/knowledge_base/)
python -m drp_system.rag.ingest_cases

# Single patient
python -m drp_system.main --patient-id PT-001

# Evaluate on clinical case suite (49 cases)
python -m drp_system.eval.run_eval --limit 3 --concurrency 1
```

Optional: use a virtualenv on top of pyenv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

| Path | Role |
|------|------|
| `drp_system/models/drp_schema.py` | PCNE output schema |
| `drp_system/llm/gemini_client.py` | Gemini + retry |
| `drp_system/rag/` | Ingest, retrieve, openFDA |
| `drp_system/agents/` | Medication, disease, orchestrator |
| `drp_system/eval/` | Benchmark vs `cases_analysis_v2_clean_en.csv` |
| `IMPLEMENTATION_PLAN.md` | Milestones and backlog |
| `implementation.md` | Full technical spec |

## Test cases

`cases_analysis_v2_clean_en.csv` holds **49** annotated cases (`PRM`: Safety, Adherence, Efficacy, Indication). The eval runner reports:

- **DRP detection rate** — model flags `drp_present`
- **PRM→PCNE match** — predicted P-category aligns with ground-truth PRM

## Rate limits

Free Gemini tier: **10 RPM**, **250 RPD**. Each patient uses **3** API calls. Use `--concurrency 1` in eval to stay within limits.

## Knowledge base

Place PDFs/CSVs in `data/knowledge_base/` (WHO formulary, DrugBank open, PCNE PDF) then:

```bash
python -m drp_system.rag.ingest
```

See [implementation.md](implementation.md) for source links.

## Disclaimer

Outputs are decision-support only and must be reviewed by a licensed clinician.
