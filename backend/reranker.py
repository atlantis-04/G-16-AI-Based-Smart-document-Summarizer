from sentence_transformers import CrossEncoder


class Reranker:
    """Rerank retrieved chunks with a cross-encoder relevance model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_with_score = dict(chunk)
            chunk_with_score["rerank_score"] = float(score)
            reranked_chunks.append(chunk_with_score)

        reranked_chunks.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
        return reranked_chunks[:top_k]
