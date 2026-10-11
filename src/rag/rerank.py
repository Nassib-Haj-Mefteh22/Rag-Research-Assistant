from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def score(self, question: str, texts: list[str]) -> list[float]:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict([(question, t) for t in texts])]
