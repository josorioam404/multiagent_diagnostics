"""
Seed ChromaDB from cases_analysis_v2_clean_en.csv (justification + evidence).

Use when full PDF knowledge base is not yet available:

    python -m drp_system.rag.ingest_cases
"""
import csv

import chromadb
from sentence_transformers import SentenceTransformer

from drp_system import config
from drp_system.rag.ingest import chunk_text


def ingest_cases_csv(path: str | None = None) -> int:
    csv_path = path or config.CASES_CSV_PATH
    all_chunks: list[str] = []
    metadata_ids: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            case_id = row.get("ID", "").strip()
            meds = row.get("LM", "")
            disease = row.get("HS", "")
            prm = row.get("PRM", "")
            jus = row.get("Jus", "")
            te = row.get("TE", "")
            text = (
                f"Case {case_id}\n"
                f"Disease: {disease}\n"
                f"Medications: {meds}\n"
                f"Problem type: {prm}\n"
                f"Analysis: {jus}\n"
                f"Clinical evidence: {te}"
            )
            for i, chunk in enumerate(
                chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            ):
                all_chunks.append(chunk)
                metadata_ids.append(f"case_{case_id}_{i}")

    if not all_chunks:
        print("No case chunks produced.")
        return 0

    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        ids = metadata_ids[i : i + batch_size]
        embeddings = embedder.encode(batch).tolist()
        collection.add(documents=batch, embeddings=embeddings, ids=ids)

    print(f"Seeded {len(all_chunks)} chunks from {csv_path}")
    return len(all_chunks)


if __name__ == "__main__":
    ingest_cases_csv()
