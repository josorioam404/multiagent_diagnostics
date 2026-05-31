import logging
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from drp_system import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_resources():
    embedder = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        collection = client.get_collection(config.COLLECTION_NAME)
    except Exception:
        logger.warning(
            "Chroma collection '%s' missing — run ingest_cases or rag.ingest",
            config.COLLECTION_NAME,
        )
        collection = client.get_or_create_collection(config.COLLECTION_NAME)
    return embedder, collection


def retrieve(query: str, top_k: int | None = None) -> list[str]:
    """Return top_k relevant text chunks for a query."""
    k = top_k or config.RAG_TOP_K
    embedder, collection = _get_resources()

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=min(k, collection.count()))
    docs = results.get("documents") or [[]]
    return docs[0] if docs else []
