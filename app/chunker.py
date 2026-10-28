from pathlib import Path
from typing import List, Dict, Any

MAX_WORDS = 400
OVERLAP_WORDS = 60


def chunk_text(text: str, source_file: str) -> List[Dict[str, Any]]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + MAX_WORDS, len(words))
        chunks.append({
            "chunk_id": chunk_id,
            "file": source_file,
            "text": " ".join(words[start:end]),
            "start_word": start,
            "end_word": end,
        })
        chunk_id += 1
        start = end - OVERLAP_WORDS
        if start >= len(words) or end >= len(words):
            break
    return chunks


def chunk_document(file_path: Path) -> List[Dict[str, Any]]:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    else:
        text = file_path.read_text(encoding="utf-8")
    return chunk_text(text, file_path.name)