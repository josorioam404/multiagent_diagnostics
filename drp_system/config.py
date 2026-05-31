import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

CHROMA_PATH = str(ROOT_DIR / "data" / "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Heavier biomedical model (uncomment when RAM allows):
# EMBED_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

KNOWLEDGE_BASE_DIR = str(ROOT_DIR / "data" / "knowledge_base")
CASEBOOK_PDF = str(ROOT_DIR / "pharmacotherapy-casebook_929.pdf")
CASES_CSV_PATH = str(ROOT_DIR / "cases_analysis_v2_clean_en.csv")

RAG_TOP_K = 5
RAG_CONTEXT_MAX_CHARS = 4000
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
GEMINI_MAX_OUTPUT_TOKENS = 8192

COLLECTION_NAME = "drp_knowledge"
