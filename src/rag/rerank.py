"""Step 7.4: cross-encoder reranker, re-sorts a short candidate list using question+chunk together.

WHY: the bi-encoder (embed.py) embeds question and chunk separately, so it is fast but crude.
A cross-encoder feeds the pair through the model at once, which is far more accurate but too
slow for all ~5k chunks, so we only use it on the ~20 candidates found by dense/hybrid search.
"""

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small (~90 MB), trained on MS MARCO


class Reranker:
    """Loads the cross-encoder on first use, so modes without reranking never pay for it."""

    def __init__(self, model_name: str = RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def score(self, question: str, texts: list[str]) -> list[float]:
        """One relevance score per text (higher = more relevant). Raw logits, not 0-1 probabilities,
        and only comparable within one question.
        """
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return [float(s) for s in self._model.predict([(question, t) for t in texts])]
