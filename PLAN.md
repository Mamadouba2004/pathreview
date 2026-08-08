# Solution plan

**Issue:** [#24 — Hybrid retriever over-weights keyword results when query contains technology names](https://github.com/ascherj/pathreview/issues/24)

**Author:** Mamadou Ba ([@Mamadouba2004](https://github.com/Mamadouba2004))
**Branch:** `fix/24-hybrid-retriever-keyword-overweight`
**Reproduction commit:** [`eaf271f`](https://github.com/Mamadouba2004/pathreview/commit/eaf271f)

---

## Understand

**What needs to change:** the score-blending step inside
`HybridRetriever.retrieve()`. Today it merges vector and BM25 results in a way
that lets keyword noise outrank semantically relevant chunks, and I need to
replace that merge with one that puts the two score families on a comparable
footing.

**Root cause.** The issue title blames the weighting, and the issue body says the
blend is "a fixed 50/50 weight." Neither is quite right, and this matters for the
fix. `__init__` actually defaults to `vector_weight=0.7, keyword_weight=0.3`. The
real problem is the *normalization*: each retriever's scores are divided by the
maximum score **within that retriever's own result set**.

That is not a normalization onto a shared scale — it preserves each family's
native dynamic range:

- `VectorStore.query()` (`rag/retriever/vector_store.py`) converts a ChromaDB
  distance to a similarity with `1 / (1 + distance)`. That function is bounded
  and compressive, so real similarities cluster in a narrow band. In my
  reproduction the eight chunks span **0.4082 – 0.5618**. Dividing by the max
  leaves the vector signal spanning only about **27%** of the 0–1 range — the
  worst chunk still scores 0.727.
- BM25 scores start at exactly 0 for any chunk that doesn't contain the query
  term. Dividing by the max therefore always stretches the keyword signal across
  the **full 0–1 range**.

So the effective weighting is inverted relative to the nominal one. Multiplying
each family's usable spread by its weight: vector contributes `0.7 × 0.27 ≈
0.191` of achievable spread, keyword contributes `0.3 × 1.00 = 0.300`. **The
keyword term controls roughly 61% of the ranking despite a nominal 30% weight.**
Re-tuning the two constants would paper over this; the normalization is what has
to change.

**Expected vs. actual.** Querying a technology name should surface the chunk that
actually describes work with that technology. Actual, measured in
`tests/unit/test_hybrid_retriever.py`: for the query `"React"` over a resume plus
an unrelated project's README, `create-react-app` boilerplate ranks first at
blended `0.9490`, ahead of the resume line describing real React work at
`0.8636`. The downstream symptom is portfolio feedback that cites the wrong
project or quotes boilerplate.

## Map

Files I expect to touch:

| File | Change |
| --- | --- |
| `rag/retriever/hybrid.py` | Primary fix. Replace the max-normalization + weighted-sum block inside `HybridRetriever.retrieve()`; likely extract a private `_fuse()` helper so the blending is testable on its own. |
| `tests/unit/test_hybrid_retriever.py` | Already added in the reproduction commit. Extend with regression tests for the fused ranking and the edge cases below. |
| `docs/ARCHITECTURE.md` | Only if the retrieval section documents the 0.7/0.3 weighting — needs checking; the constructor signature may become part of the documented contract. |

Files I expect to **read but not modify**:

- `rag/retriever/vector_store.py` — to confirm `query()` really is the only
  producer of the `score` field and that `1 / (1 + distance)` is the only
  transform applied.
- `rag/retriever/keyword_search.py` — to confirm `search()` is the only producer
  of `bm25_score` and that it returns results already sorted descending.
- Callers of `HybridRetriever`. A repo-wide search currently finds the class
  referenced only in `rag/retriever/hybrid.py`, so no caller should break — but I
  will re-run that search before changing the constructor signature.

## Plan

1. **Extract the blending logic into a testable unit.** Pull the normalize-and-
   sum block out of `HybridRetriever.retrieve()` into a private method that takes
   the two result lists and returns fused, sorted results. No behaviour change in
   this step — the existing reproduction test must still fail identically, which
   proves the extraction was faithful.
2. **Replace per-retriever max-normalization with Reciprocal Rank Fusion.**
   Score each chunk as `Σ weight / (k + rank)` over the retrievers that returned
   it, with `k = 60` as the standard constant. RRF consumes only the rank order,
   so the compressed dynamic range of `1 / (1 + distance)` stops mattering and
   `vector_weight` / `keyword_weight` regain their literal meaning. I'm choosing
   RRF over min-max or z-score normalization because both of those still depend
   on the score distribution of one query's candidate set, which is exactly the
   fragility causing this bug.
3. **Re-scale the output so `min_score` keeps meaning something.** Raw RRF scores
   are small (order 1/60) and would put every chunk below the default
   `min_score = 0.3`, silently emptying the result set. Normalize the fused
   scores back onto 0–1 before the threshold filter, and confirm against the
   `min_score` default in the `retrieve()` signature.
4. **Make the keyword-only path stop free-riding.** A chunk returned by only one
   retriever should be penalized for the other's silence rather than scored as if
   the other retriever gave it zero. Under RRF this falls out naturally, but I'll
   assert it explicitly so a future refactor can't regress it.
5. **Turn the reproduction test green and add regression coverage.** Flip
   `test_technology_name_query_ranks_relevant_chunk_first` from failing to
   passing without weakening the assertion, add the edge cases below, then run
   `make check && make test-unit`.

## Inputs & outputs

**Inputs** to the changed code path — unchanged from today:

- `query: str`, `profile_id: str`, `query_embedding: list[float]`,
  `max_chunks: int = 10`, `min_score: float = 0.3` on `retrieve()`.
- `vector_results: list[dict]` from `VectorStore.query()`, each with keys
  `id`, `text`, `metadata`, `score` (a `1/(1+distance)` similarity).
- `keyword_results: list[dict]` from `KeywordSearcher.search()`, each a copy of
  the original chunk plus `bm25_score: float`.

**Outputs** — the dict shape stays identical so nothing downstream breaks:

- `list[dict]` with `id`, `text`, `metadata`, `score`, `vector_score`,
  `keyword_score`, sorted by `score` descending and truncated to `max_chunks`.

**What changes:** the numeric meaning of `score`, `vector_score` and
`keyword_score`. They become rank-derived contributions rather than
max-normalized raw scores. `score` remains in 0–1 so the `min_score` contract
holds. **What does not change:** the method signature, the key names, the sort
order guarantee, or the truncation behaviour.

## Risks & unknowns

- **`min_score` becomes meaningless if I forget step 3.** Raw RRF values live
  around 0.016; the `min_score = 0.3` default in `retrieve()` would filter out
  every chunk and return an empty list, which the caller would read as "no
  evidence found." *Investigation:* assert on a populated result set with the
  default `min_score` before and after the change.
- **`vector_score` / `keyword_score` may be consumed somewhere I haven't found.**
  I only searched for the class name `HybridRetriever`, not for these dict keys.
  If a UI or logging path reads them as raw similarities, rank-derived values
  would be misleading. *Investigation:* grep for `vector_score` and
  `keyword_score` across `api/`, `rag/`, and `frontend/src/` before finalizing.
- **`_get_all_chunks()` looks like dead code, and I don't know why.** In
  `retrieve()` its return value is assigned to `all_chunks` and then never used —
  `self.keyword_searcher.search()` is called without the searcher ever being
  indexed inside `retrieve()`. Either indexing happens elsewhere in the ingestion
  path, or keyword search silently returns `[]` in production (the empty-index
  guard in `KeywordSearcher.search()` returns an empty list). If the latter, the
  bug in production may look different from my reproduction. *Investigation:*
  trace where `KeywordSearcher.index()` is called from before I write the fix;
  this is the first thing I'll check.
- **RRF may be the wrong call if the maintainer wants tunable weights.** RRF
  deliberately discards score magnitude. If someone downstream depends on
  "how much better" the top chunk is, min-max normalization over the union of
  candidates would be the more conservative fix. *Mitigation:* raise this in the
  PR description and keep the fusion behind one extractable method so swapping it
  is a small diff.
- **No existing test file for this module.** `tests/unit/` had no
  `test_hybrid_retriever.py` before my reproduction commit, so I have no
  established fixtures for this class and may be diverging from a convention I
  can't see. *Investigation:* compare against `tests/unit/test_keyword_search.py`,
  which is the closest neighbour.

## Edge cases

The fix must handle each of these gracefully:

1. **A chunk returned by the keyword searcher but absent from the vector
   results.** Verified in the reproduction: it gets `vector_score = 0.0`, so its
   blended score is exactly `keyword_weight` = `0.30`, which meets the default
   `min_score = 0.3` and passes the quality filter. A chunk with no measured
   semantic relevance at all currently clears the quality floor on a boundary
   coincidence. After the fix it must rank below any chunk both retrievers found.
2. **A chunk returned by the vector store but absent from the keyword results.**
   The mirror case, and common: a semantically relevant chunk that happens not to
   contain the literal query token. It must not be penalized into oblivion — this
   is the case a naive "require both retrievers" fix would break.
3. **Both retrievers return empty lists.** `retrieve()` must return `[]` rather
   than raising. Today `max(..., default=1.0)` guards the division; whatever
   replaces it needs its own guard against an empty candidate set.
4. **Every BM25 score is 0.0.** Happens when the query term appears in roughly
   half or more of the corpus — BM25Okapi's IDF goes to zero. I hit this while
   building the reproduction with a 4-chunk corpus. Dividing by
   `keyword_scores_max = 0` is already guarded, but the fused path must degrade
   cleanly to vector-only ranking rather than collapsing every keyword
   contribution into a tie that reshuffles the vector order.
5. **A single-chunk result set.** With one candidate, max-normalization makes its
   normalized score exactly 1.0 and any rank-based scheme makes it rank 1. The
   result must still respect `min_score` rather than being auto-admitted.

---

*Living document — I expect the `_get_all_chunks()` finding in particular to
change this plan during Week 9.*
