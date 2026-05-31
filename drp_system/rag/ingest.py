"""
Build the ChromaDB vector store from PDFs/CSVs in data/knowledge_base/.

    python -m drp_system.rag.ingest
"""
import csv
import os

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

from drp_system import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def ingest_pdf(path: str) -> list[str]:
    doc = fitz.open(path)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return chunk_text(full_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)


def ingest_csv(path: str) -> list[str]:
    chunks: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", row.get("name", ""))
            description = row.get("Description", row.get("description", ""))
            indication = row.get("Indication", row.get("indication", ""))
            interactions = row.get(
                "Drug Interactions", row.get("drug_interactions", "")
            )
            text = (
                f"Drug: {name}\n"
                f"Description: {description}\n"
                f"Indication: {indication}\n"
                f"Interactions: {interactions}"
            )
            if name or description:
                chunks.extend(
                    chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                )
    return chunks


def _ensure_casebook_in_kb(kb_dir: str) -> None:
    """Symlink project casebook PDF into knowledge_base when present."""
    casebook = config.CASEBOOK_PDF
    link_path = os.path.join(kb_dir, os.path.basename(casebook))
    if os.path.isfile(casebook) and not os.path.exists(link_path):
        os.symlink(os.path.relpath(casebook, kb_dir), link_path)


def build_vector_store(reset: bool = False) -> int:
    """Ingest knowledge_base files into ChromaDB. Returns chunk count."""
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not os.path.isdir(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)
    _ensure_casebook_in_kb(kb_dir)

    all_chunks: list[str] = []
    for fname in sorted(os.listdir(kb_dir)):
        fpath = os.path.join(kb_dir, fname)
        if not os.path.isfile(fpath):
            continue
        print(f"Ingesting: {fname}")
        if fname.endswith(".pdf"):
            all_chunks.extend(ingest_pdf(fpath))
        elif fname.endswith(".csv") and fname != "cases_seed.csv":
            all_chunks.extend(ingest_csv(fpath))

    if not all_chunks:
        print("No chunks from knowledge_base — run ingest_cases or add PDFs/CSVs.")
        return 0

    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings = embedder.encode(batch).tolist()
        ids = [f"kb_{i + j}" for j in range(len(batch))]
        collection.add(documents=batch, embeddings=embeddings, ids=ids)
        print(f"  Stored {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")

    print(f"Vector store built: {len(all_chunks)} chunks.")
    return len(all_chunks)


if __name__ == "__main__":
    build_vector_store(reset=True)
