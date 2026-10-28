from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts):
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def get_model_version() -> str:
    return MODEL_NAME