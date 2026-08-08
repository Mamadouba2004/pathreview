"""Tests for hybrid.py

Regression suite for issue #24 (hybrid retriever over-weights keyword
results when the query contains a technology name).

Original root cause: HybridRetriever.retrieve normalized each retriever's
scores by the maximum score *within that retriever's own result set* before
summing them. Vector similarities from VectorStore.query (1/(1+distance))
cluster in a narrow band, so normalizing them onto 0-1 left the vector
signal spanning a fraction of the range, while BM25 scores start at 0 and
always spanned the full range after normalization - so the nominal 0.7/0.3
weighting was not the effective weighting, and keyword noise could outrank
semantically relevant chunks.

Fix: replace the per-retriever normalize-then-sum with weighted Reciprocal
Rank Fusion (RRF), which consumes only each retriever's rank position and
is therefore unaffected by the two retrievers having different score
distributions. See rag/retriever/hybrid.py for the implementation and
docs/PLAN.md (on this branch) for the full write-up.
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
    """Stands in for a ChromaDB collection."""

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
    """Regression suite for issue #24."""

    @pytest.fixture
    def searcher(self):
        searcher = KeywordSearcher()
        searcher.index(CORPUS)
        return searcher

    @pytest.fixture
    def retriever(self, searcher):
        return HybridRetriever(FakeVectorStore(), searcher)

    def test_vector_scores_occupy_a_narrower_range_than_bm25_scores(self, searcher):
        """Documents why per-retriever normalization (the old approach) was fragile.

        This is the mechanism behind issue #24: normalizing each retriever's
        scores independently before combining them is unsafe precisely
        because their native ranges differ this much. It's why the fix uses
        rank-based fusion instead of re-normalizing raw scores.
        """
        similarities = [1 / (1 + d) for d in DISTANCES.values()]
        vector_spread = (max(similarities) - min(similarities)) / max(similarities)

        bm25 = [r["bm25_score"] for r in searcher.search("React", top_k=10)]
        keyword_spread = (max(bm25) - min(bm25)) / max(bm25)

        assert keyword_spread == pytest.approx(1.0)
        assert vector_spread < 0.5

    def test_technology_name_query_ranks_relevant_chunk_first(self, retriever):
        """The core regression test for issue #24.

        Querying a technology name must surface the resume line that
        actually describes React work, not create-react-app boilerplate
        from an unrelated project's README.
        """
        results = retriever.retrieve(query="React", profile_id="p1", query_embedding=[0.0] * 8)

        assert results[0]["id"] == "resume-3", (
            f"expected 'resume-3' at rank 1, got {results[0]['id']!r} "
            f"(score={results[0]['score']:.4f})"
        )

    def test_score_equals_sum_of_vector_and_keyword_contributions(self, retriever):
        """score, vector_score, and keyword_score are consistent with each other.

        Under the old normalize-then-sum approach, the weights were baked
        into `score` but not into the returned `vector_score` /
        `keyword_score` fields, so a caller could not reconstruct `score`
        from the two breakdown fields. The fused implementation applies
        weights before normalizing, so the breakdown is directly additive.
        """
        results = retriever.retrieve(query="React", profile_id="p1", query_embedding=[0.0] * 8)

        assert results, "expected at least one result"
        for r in results:
            assert r["score"] == pytest.approx(r["vector_score"] + r["keyword_score"])

    def test_chunk_absent_from_one_retriever_gets_zero_from_it(self, searcher):
        """Covers PLAN.md edge cases 1 and 2: a chunk found by only one retriever.

        A chunk missing from the vector results must show vector_score == 0
        (not be penalized further, not be assumed average); the mirror case
        holds for a chunk missing from the keyword results. Both must still
        appear in the final results - a single-retriever hit is a real
        signal, just a partial one.
        """
        top_keyword_id = searcher.search("React", top_k=10)[0]["id"]

        class NarrowVectorStore(FakeVectorStore):
            """Vector store that never returns the top keyword hit."""

            def query(self, query_embedding, collection_name, n_results=10):
                base = super().query(query_embedding, collection_name, n_results=99)
                return [r for r in base if r["id"] != top_keyword_id][:n_results]

        retriever = HybridRetriever(NarrowVectorStore(), searcher)
        results = retriever.retrieve(
            query="React", profile_id="p1", query_embedding=[0.0] * 8, max_chunks=10
        )

        keyword_only = next(r for r in results if r["id"] == top_keyword_id)
        assert keyword_only["vector_score"] == 0.0
        assert keyword_only["keyword_score"] > 0.0

        # Mirror case: any chunk absent from the keyword results (every
        # chunk in this corpus except the top-10 BM25 matches - here, all
        # of them are indexed, so this checks the ones ranked outside the
        # keyword searcher's own top_k window) must show keyword_score == 0.
        keyword_result_ids = {r["id"] for r in searcher.search("React", top_k=10)}
        vector_only_present = [r for r in results if r["id"] not in keyword_result_ids]
        for r in vector_only_present:
            assert r["keyword_score"] == 0.0
            assert r["vector_score"] > 0.0

    def test_empty_results_from_both_retrievers_returns_empty_list(self):
        """Covers PLAN.md edge case 3: no candidates from either retriever.

        Must return [] rather than raising (e.g. a ZeroDivisionError from
        dividing by a max score computed over an empty set).
        """

        class EmptyVectorStore(FakeVectorStore):
            def query(self, query_embedding, collection_name, n_results=10):
                return []

        empty_searcher = KeywordSearcher()  # never indexed -> search() returns []
        retriever = HybridRetriever(EmptyVectorStore(), empty_searcher)

        results = retriever.retrieve(query="React", profile_id="p1", query_embedding=[0.0] * 8)

        assert results == []

    def test_single_candidate_does_not_raise_and_is_bounded(self):
        """Covers PLAN.md edge case 5: exactly one candidate in the batch.

        A lone candidate is trivially its own batch maximum, so it will
        always normalize to score == 1.0 and pass any min_score - this
        was true before the fix too. What the fix must guarantee is that
        this case doesn't divide by zero or otherwise raise.
        """
        chunk = {"id": "only-one", "text": "irrelevant content", "metadata": {}}

        class OneChunkVectorStore(FakeVectorStore):
            def query(self, query_embedding, collection_name, n_results=10):
                return [{**chunk, "score": 0.42}]

        searcher = KeywordSearcher()
        searcher.index([chunk])
        retriever = HybridRetriever(OneChunkVectorStore(), searcher)

        results = retriever.retrieve(
            query="query terms not present in the chunk",
            profile_id="p1",
            query_embedding=[0.0] * 8,
        )

        assert len(results) == 1
        assert 0.0 <= results[0]["score"] <= 1.0

    def test_tied_bm25_scores_do_not_override_vector_ranking_at_default_weights(self):
        """Covers PLAN.md edge case 4: every BM25 score ties (query term is
        in every chunk, so BM25's IDF component collapses toward zero).

        KeywordSearcher.search still returns top_k chunks in this case
        (Python's sort is stable, so ties keep their original order) - RRF
        consumes that position regardless of the tied score's magnitude.
        At the default weights (vector_weight=0.7 > keyword_weight=0.3)
        the dominant vector signal keeps the fused ranking aligned with the
        vector-only ranking even when the keyword ranking is meaningless.

        This is a real, acknowledged limitation of pure RRF - it is blind to
        score magnitude - flagged in the PR as a residual risk rather than
        silently assumed away.
        """
        tied_corpus = [
            {"id": "a", "text": "python python python", "metadata": {}},
            {"id": "b", "text": "python python python", "metadata": {}},
            {"id": "c", "text": "python python python", "metadata": {}},
        ]

        class TiedVectorStore(FakeVectorStore):
            def query(self, query_embedding, collection_name, n_results=10):
                order = ["c", "b", "a"]
                scores = {"c": 0.9, "b": 0.7, "a": 0.5}
                return [
                    {
                        "id": cid,
                        "text": next(x["text"] for x in tied_corpus if x["id"] == cid),
                        "metadata": {},
                        "score": scores[cid],
                    }
                    for cid in order
                ][:n_results]

        searcher = KeywordSearcher()
        searcher.index(tied_corpus)
        bm25_scores = {r["bm25_score"] for r in searcher.search("python", top_k=10)}
        assert len(bm25_scores) == 1, "expected every BM25 score to tie for this corpus"

        retriever = HybridRetriever(TiedVectorStore(), searcher)
        results = retriever.retrieve(
            query="python", profile_id="p1", query_embedding=[0.0] * 4, min_score=0.0
        )

        assert [r["id"] for r in results] == ["c", "b", "a"]
