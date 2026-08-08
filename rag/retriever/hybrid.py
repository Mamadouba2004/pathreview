"""Hybrid retriever combining vector and keyword search."""

import structlog

from .keyword_search import KeywordSearcher
from .vector_store import VectorStore

logger = structlog.get_logger()

# Reciprocal Rank Fusion constant. 60 is the value used in the original RRF
# paper (Cormack et al., 2009) and is the de facto default in most hybrid
# search implementations; it flattens the influence of very top-ranked
# results without needing to be tuned per corpus.
_RRF_K = 60


class HybridRetriever:
    """Combines vector similarity and BM25 keyword search.

    Each retriever's ranked list is fused using Reciprocal Rank Fusion (RRF)
    rather than by normalizing each retriever's raw scores independently.
    RRF only consumes rank position, so it is unaffected by the two
    retrievers having very different score distributions - which is what
    previously let keyword matches dominate rankings (see issue #24).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_searcher: KeywordSearcher,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: VectorStore instance
            keyword_searcher: KeywordSearcher instance
            vector_weight: Weight for vector scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        self.vector_store = vector_store
        self.keyword_searcher = keyword_searcher
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        profile_id: str,
        query_embedding: list[float],
        max_chunks: int = 10,
        min_score: float = 0.3,
    ) -> list[dict]:
        """Retrieve chunks using hybrid approach.

        Args:
            query: Text query
            profile_id: Profile identifier for collection selection
            query_embedding: Embedding vector for query
            max_chunks: Maximum chunks to return
            min_score: Minimum score threshold (0-1)

        Returns:
            List of dicts with blended scores
        """
        collection_name = f"profile_{profile_id}"

        # Vector search
        vector_results = self.vector_store.query(
            query_embedding, collection_name, n_results=max_chunks * 2
        )

        # Keyword search
        keyword_results = self.keyword_searcher.search(query, top_k=max_chunks * 2)

        fused = self._fuse_rankings(vector_results, keyword_results)

        if not fused:
            logger.info(
                "hybrid_retrieval_complete",
                query_len=len(query),
                vector_results=len(vector_results),
                keyword_results=len(keyword_results),
                blended_count=0,
                filtered_count=0,
                final_count=0,
            )
            return []

        # Normalize the fused (rank-derived) scores onto 0-1 using the best
        # score achieved in *this* result set, so min_score keeps meaning.
        # Unlike the old per-retriever normalization, this happens after
        # fusion, on a single unified score - it can no longer let one
        # retriever's compressed score range be stretched to look stronger
        # than the other's.
        max_fused_score = max(entry["raw_score"] for entry in fused.values())

        blended = {}
        for chunk_id, entry in fused.items():
            if max_fused_score > 0:
                score = entry["raw_score"] / max_fused_score
                vector_score = entry["vector_contribution"] / max_fused_score
                keyword_score = entry["keyword_contribution"] / max_fused_score
            else:
                score = vector_score = keyword_score = 0.0

            blended[chunk_id] = {
                "id": chunk_id,
                "text": entry["text"],
                "metadata": entry["metadata"],
                "score": score,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
            }

        # Filter by min_score and sort
        results = [r for r in blended.values() if r["score"] >= min_score]
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top max_chunks
        final_results = results[:max_chunks]

        logger.info(
            "hybrid_retrieval_complete",
            query_len=len(query),
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            blended_count=len(blended),
            filtered_count=len(results),
            final_count=len(final_results),
        )

        return final_results

    def _fuse_rankings(self, vector_results: list[dict], keyword_results: list[dict]) -> dict:
        """Fuse two ranked result lists with weighted Reciprocal Rank Fusion.

        Each chunk's contribution from a retriever is
        ``weight / (_RRF_K + rank + 1)``, where rank is the chunk's 0-indexed
        position in that retriever's own ranked output. A chunk missing from
        a retriever's results contributes 0 from that retriever - it is not
        assigned that retriever's lowest possible score, and it is not
        assumed to be equally good as the retriever's top hit either.

        Args:
            vector_results: Ranked output of VectorStore.query, best first.
            keyword_results: Ranked output of KeywordSearcher.search, best first.

        Returns:
            Dict keyed by chunk id, each value holding the chunk's text,
            metadata, combined raw_score, and the separate vector/keyword
            contributions that produced it.
        """
        fused: dict = {}

        for rank, result in enumerate(vector_results):
            chunk_id = result["id"]
            entry = fused.setdefault(
                chunk_id,
                {
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "raw_score": 0.0,
                    "vector_contribution": 0.0,
                    "keyword_contribution": 0.0,
                },
            )
            contribution = self.vector_weight / (_RRF_K + rank + 1)
            entry["raw_score"] += contribution
            entry["vector_contribution"] += contribution

        for rank, result in enumerate(keyword_results):
            chunk_id = result.get("id", "")
            entry = fused.setdefault(
                chunk_id,
                {
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "raw_score": 0.0,
                    "vector_contribution": 0.0,
                    "keyword_contribution": 0.0,
                },
            )
            contribution = self.keyword_weight / (_RRF_K + rank + 1)
            entry["raw_score"] += contribution
            entry["keyword_contribution"] += contribution

        return fused
