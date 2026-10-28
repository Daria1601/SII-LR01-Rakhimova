# RAG-ассистент

Корпус: кадровые документы (политика отпусков, FAQ, onboarding).

## Запуск
1. `python -m venv venv`
2. Активация: Windows — `venv\Scripts\activate`, Linux/macOS — `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python scripts/build_index.py`
5. `uvicorn app.main:app --reload --port 8000`
6. `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"...\"}"`