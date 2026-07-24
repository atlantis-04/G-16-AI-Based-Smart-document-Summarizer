
from rank_bm25 import BM25Okapi

from backend.retriever import ChromaRetriever


class HybridRetriever:
    """Combines ChromaDB semantic search with BM25 keyword scoring."""

    def __init__(self, semantic_weight: float = 0.5, collection_name: str = "rag_docs"):
        self.chroma = ChromaRetriever(collection_name=collection_name)
        self.semantic_weight = semantic_weight
        self.bm25_weight = 1.0 - semantic_weight
        self.bm25_index = None
        self.bm25_docs = []

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _build_bm25_index(self, docs: list[str]):
        self.bm25_docs = docs
        tokenized_docs = [self._tokenize(doc) for doc in docs]
        self.bm25_index = BM25Okapi(tokenized_docs) if tokenized_docs else None
        return self.bm25_index

    def _get_all_docs_from_chroma(self) -> list[str]:
        results = self.chroma.collection.get()
        return results.get("documents") or []

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        if self.collection_count() == 0:
            return []

        semantic_results = self.chroma.retrieve(query, n_results=n_results * 2)
        if not semantic_results:
            return []

        all_docs = self._get_all_docs_from_chroma()
        bm25_index = self._build_bm25_index(all_docs)
        bm25_scores = bm25_index.get_scores(self._tokenize(query)) if bm25_index else []
        max_bm25_score = max(bm25_scores) if len(bm25_scores) else 0.0

        scored_results = []
        for result in semantic_results:
            semantic_score = max(0.0, 1.0 - float(result.get("distance", 0.0)))
            bm25_raw_score = 0.0

            try:
                doc_index = all_docs.index(result["text"])
                bm25_raw_score = float(bm25_scores[doc_index])
            except (ValueError, IndexError):
                bm25_raw_score = 0.0

            bm25_score = bm25_raw_score / max_bm25_score if max_bm25_score > 0 else 0.0
            final_score = (
                self.semantic_weight * semantic_score
                + self.bm25_weight * bm25_score
            )

            scored_results.append(
                {
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "distance": result.get("distance", 0.0),
                    "semantic_score": semantic_score,
                    "bm25_score": bm25_score,
                    "final_score": final_score,
                }
            )

        scored_results.sort(key=lambda item: item["final_score"], reverse=True)
        return scored_results[:n_results]

    def collection_count(self) -> int:
        return self.chroma.collection_count()
