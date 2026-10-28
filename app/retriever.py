from pathlib import Path
from typing import List, Dict, Any
import chromadb
from app.embedder import encode

CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "documents"
SIMILARITY_THRESHOLD = 0.35


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str, top_k: int = 2) -> List[Dict[str, Any]]:
    collection = get_collection()
    query_embedding = encode([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    retrieved = []
    if not results["ids"] or not results["ids"][0]:
        return retrieved
    for i, _ in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity = 1.0 - distance
        retrieved.append({
            "file": metadata["file"],
            "chunk_id": metadata["chunk_id"],
            "score": round(float(similarity), 4),
            "text": results["documents"][0][i],
        })
    return retrieved