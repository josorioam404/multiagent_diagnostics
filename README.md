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

# Build vector store from Pharmacotherapy Casebook (place PDF in repo root)
python -m drp_system.rag.ingest

# Optional: also seed from eval CSV (dev only — can inflate benchmark scores)
# python -m drp_system.rag.ingest_cases

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

Primary source: **Pharmacotherapy Casebook** (`pharmacotherapy-casebook_929.pdf` in the project root). Ingest links it into `data/knowledge_base/` and embeds into ChromaDB:

```bash
python -m drp_system.rag.ingest
```

The PDF is gitignored (copyright). Additional PDFs/CSVs can be dropped in `data/knowledge_base/`.

## Disclaimer

Outputs are decision-support only and must be reviewed by a licensed clinician.
