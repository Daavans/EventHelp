"""Retrieval over the festival FAQ: split, embed, cache and search."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.llm import EMBEDDING_MODEL, embed

DATA_DIR = Path(__file__).parent.parent / "data"
FAQ_PATH = DATA_DIR / "faq.md"
CACHE_PATH = DATA_DIR / "embeddings_cache.npz"


def load_chunks(path: Path = FAQ_PATH) -> list[str]:
    """Split the FAQ into chunks: one per question ("## ...") including its answer."""
    text = path.read_text(encoding="utf-8")
    sections = text.split("\n## ")[1:]  # [0] is the title and intro, not a question
    return ["## " + section.strip() for section in sections]


def load_embeddings(chunks: list[str]) -> np.ndarray:
    """Return one embedding per chunk, from the cache file if the FAQ has not changed."""
    # The hash changes when the FAQ text or the embedding model changes, invalidating the cache.
    faq_hash = hashlib.sha256((EMBEDDING_MODEL + "\n".join(chunks)).encode("utf-8")).hexdigest()

    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        if str(cache["faq_hash"]) == faq_hash:
            return cache["vectors"]

    print(f"Embedding {len(chunks)} FAQ chunks...")
    vectors = np.array(embed(chunks, task_type="RETRIEVAL_DOCUMENT"))
    np.savez(CACHE_PATH, vectors=vectors, faq_hash=faq_hash)
    return vectors


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Similarity between the query vector and every row of the matrix (1 = same direction)."""
    return (matrix @ query) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))


@lru_cache(maxsize=1)
def _load_index() -> tuple[list[str], np.ndarray]:
    """Load chunks and embeddings once per process."""
    chunks = load_chunks()
    return chunks, load_embeddings(chunks)


def search(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Return the k FAQ chunks most similar to the query, with their similarity scores."""
    chunks, vectors = _load_index()
    query_vector = np.array(embed([query], task_type="RETRIEVAL_QUERY")[0])
    scores = cosine_similarity(query_vector, vectors)
    best = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best]


if __name__ == "__main__":
    examples = [
        "Can I bring my own food?",
        "How much does a beer cost?",
        "Is there a bus from the station?",
        "Mag mijn zoontje van 14 mee?",
        "What is the capital of France?",
        "What is the meaning of life?",
        "What is the airspeed velocity of an unladen swallow?",
        "What is the best way to get to the festival?",
    ]
    for question in examples:
        print(f"\nQ: {question}")
        for chunk, score in search(question):
            print(f"  {score:.3f}  {chunk.splitlines()[0]}")
