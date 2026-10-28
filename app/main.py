from fastapi import FastAPI
from pydantic import BaseModel

from app.retriever import retrieve, SIMILARITY_THRESHOLD
from app.embedder import get_model_version

app = FastAPI(title="RAG Assistant")


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    file: str
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    model_version: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        return AskResponse(
            answer="Не нашёл информации по вашему вопросу",
            sources=[],
            model_version=get_model_version(),
        )
    retrieved = retrieve(question, top_k=2)
    relevant = [r for r in retrieved if r["score"] >= SIMILARITY_THRESHOLD]
    if not relevant:
        return AskResponse(
            answer="Не нашёл информации по вашему вопросу",
            sources=[],
            model_version=get_model_version(),
        )
    context_parts = []
    sources = []
    for i, item in enumerate(relevant):
        context_parts.append(f"[Фрагмент {i+1}] {item['text']}")
        sources.append(SourceItem(
            file=item["file"],
            chunk_id=item["chunk_id"],
            score=item["score"],
        ))
    context = "\n\n".join(context_parts)
    return AskResponse(
        answer=f"Согласно документам: {context}",
        sources=sources,
        model_version=get_model_version(),
    )