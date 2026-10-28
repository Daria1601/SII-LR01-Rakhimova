import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from app.chunker import chunk_document
from app.embedder import encode, get_model_version

DATA_DIR = Path("data")
CHROMA_DIR = Path(".chroma")
COLLECTION_NAME = "documents"


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    all_chunks = []
    for file_path in sorted(DATA_DIR.iterdir()):
        if file_path.suffix.lower() not in (".pdf", ".txt", ".md"):
            continue
        print(f"Обработка: {file_path.name}")
        all_chunks.extend(chunk_document(file_path))
    if not all_chunks:
        print("Нет документов для индексации.")
        return
    texts = [c["text"] for c in all_chunks]
    embeddings = encode(texts)
    ids = [f"{c['file']}_chunk_{c['chunk_id']}" for c in all_chunks]
    metadatas = [{"file": c["file"], "chunk_id": c["chunk_id"]} for c in all_chunks]
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
    print(f"Индекс построен: {len(all_chunks)} чанков, модель {get_model_version()}")


if __name__ == "__main__":
    main()