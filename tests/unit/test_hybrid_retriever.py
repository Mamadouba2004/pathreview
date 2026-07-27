"""Tests for hybrid.py

Reproduction for issue #24: the hybrid retriever over-weights keyword results
when the query contains a technology name.

Root cause under test: HybridRetriever.retrieve normalizes each retriever's
scores by the maximum score *within that retriever's own result set*. Vector
similarities produced by VectorStore.query are 1/(1+distance), which clusters
them into a narrow band, so max-normalization leaves the vector signal spanning
only a fraction of 0-1. BM25 scores start at 0, so max-normalization always
spreads the keyword signal across the full 0-1 range. The nominal 0.7/0.3
weighting is therefore not the effective weighting, and keyword noise wins.

test_technology_name_query_ranks_relevant_chunk_first is EXPECTED TO FAIL until
issue #24 is fixed. It is the red half of the red/green cycle.
"""

import pytest

from rag.retriever.hybrid import HybridRetriever
from rag.retriever.keyword_search import KeywordSearcher


# Two source documents for one profile: a resume, and the README of an
# unrelated side project whose boilerplate repeats "React" many times.
CORPUS = [
    {
        "id": "resume-3",
        "text": (
            "Built and shipped a React dashboard for a logistics client, owning "
            "the component architecture and state management across twelve screens."
        ),
        "metadata": {"source_id": "resume.pdf"},
    },
    {
        "id": "readme-boiler-1",
        "text": (
            "This project was bootstrapped with Create React App. Run npm start "
            "to run the React app in development mode. React will reload the page "
            "when you edit. See the React documentation on the React website."
        ),
        "metadata": {"source_id": "todo-app/README.md"},
    },
    {
        "id": "resume-1",
        "text": "Software engineering intern, summer 2025. Python and SQL pipelines.",
        "metadata": {"source_id": "resume.pdf"},
    },
    {
        "id": "resume-2",
        "text": "Bachelor of Science in Computer Information Systems, expected 2028.",
        "metadata": {"source_id": "resume.pdf"},
    },
    {
        "id": "readme-arch-1",
        "text": "Architecture overview: a Flask API backed by a Postgres database.",
        "metadata": {"source_id": "todo-app/README.md"},
    },
    {
        "id": "readme-setup-1",
        "text": "Setup: clone the repository, install dependencies, copy the env file.",
        "metadata": {"source_id": "todo-app/README.md"},
    },
    {
        "id": "readme-lic-1",
        "text": "Licensed under MIT. Contributions welcome via pull request.",
        "metadata": {"source_id": "todo-app/README.md"},
    },
    {
        "id": "resume-4",
        "text": "Volunteer tutor for introductory programming, two semesters.",
        "metadata": {"source_id": "resume.pdf"},
    },
]

# Stand-in ChromaDB distances. resume-3 is the closest semantic match to a
# question about React work; the boilerplate chunk is second but not close.
DISTANCES = {
    "resume-3": 0.78,
    "readme-boiler-1": 0.92,
    "resume-1": 1.05,
    "readme-arch-1": 1.20,
    "resume-2": 1.30,
    "readme-setup-1": 1.35,
    "resume-4": 1.40,
    "readme-lic-1": 1.45,
}


class FakeCollection:
    """Stands in for a ChromaDB collection for _get_all_chunks."""

    def get(self, include=None):
        return {
            "ids": [c["id"] for c in CORPUS],
            "documents": [c["text"] for c in CORPUS],
            "metadatas": [c["metadata"] for c in CORPUS],
        }


class FakeVectorStore:
    """Stands in for VectorStore, reproducing its 1/(1+distance) scoring."""

    def get_collection(self, name):
        return FakeCollection()

    def query(self, query_embedding, collection_name, n_results=10):
        results = [
            {
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"],
                "score": 1 / (1 + DISTANCES[c["id"]]),
            }
            for c in CORPUS
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:n_results]


@pytest.mark.unit
class TestHybridRetrieverKeywordOverWeighting:
    """Reproduction suite for issue #24."""

    @pytest.fixture
    def searcher(self):
        searcher = KeywordSearcher()
        searcher.index(CORPUS)
        return searcher

    @pytest.fixture
    def retriever(self, searcher):
        return HybridRetriever(FakeVectorStore(), searcher)

    def test_vector_scores_occupy_a_narrower_range_than_bm25_scores(self, searcher):
        """Document the precondition: the two score families have different ranges.

        This is the mechanism behind issue #24 and it passes today - it asserts
        what the code currently does, not what it should do.
        """
        similarities = [1 / (1 + d) for d in DISTANCES.values()]
        vector_spread = (max(similarities) - min(similarities)) / max(similarities)

        bm25 = [r["bm25_score"] for r in searcher.search("React", top_k=10)]
        keyword_spread = (max(bm25) - min(bm25)) / max(bm25)

        # After per-retriever max-normalization the keyword signal spans the full
        # 0-1 range while the vector signal spans well under half of it.
        assert keyword_spread == pytest.approx(1.0)
        assert vector_spread < 0.5

    def test_keyword_only_chunk_clears_min_score_on_keyword_weight_alone(self, searcher):
        """A chunk with no vector score at all still passes the quality floor.

        A chunk absent from the vector result set gets vector_score = 0.0, so its
        blended score is keyword_weight * keyword_score. The top keyword hit is
        always normalized to 1.0, giving exactly keyword_weight (0.3), which meets
        the default min_score of 0.3.
        """
        retriever = HybridRetriever(FakeVectorStore(), searcher)
        top_keyword_id = searcher.search("React", top_k=10)[0]["id"]

        # Restrict the vector store so the top keyword hit is not among its results.
        class NarrowVectorStore(FakeVectorStore):
            def query(self, query_embedding, collection_name, n_results=10):
                base = super().query(query_embedding, collection_name, n_results=99)
                return [r for r in base if r["id"] != top_keyword_id][:n_results]

        retriever.vector_store = NarrowVectorStore()
        results = retriever.retrieve(
            query="React", profile_id="p1", query_embedding=[0.0] * 8, max_chunks=10
        )

        orphan = next(r for r in results if r["id"] == top_keyword_id)
        assert orphan["vector_score"] == 0.0
        assert orphan["score"] == pytest.approx(retriever.keyword_weight)
        assert orphan["score"] >= 0.3  # clears the default min_score floor

    def test_technology_name_query_ranks_relevant_chunk_first(self, retriever):
        """EXPECTED TO FAIL until issue #24 is fixed.

        Querying a technology name should surface the resume line that actually
        describes React work, not create-react-app boilerplate from an unrelated
        project's README.
        """
        results = retriever.retrieve(
            query="React", profile_id="p1", query_embedding=[0.0] * 8
        )

        assert results[0]["id"] == "resume-3", (
            "issue #24: keyword noise outranks the semantically relevant chunk. "
            f"Got {results[0]['id']!r} at rank 1 "
            f"(blended={results[0]['score']:.4f}); expected 'resume-3'."
        )
