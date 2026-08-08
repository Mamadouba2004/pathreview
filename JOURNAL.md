# PathReview — Module 3 Contribution Journal

**Contributor:** Mamadou Ba ([@Mamadouba2004](https://github.com/Mamadouba2004))
**Course:** CodePath AI201 — Applications of AI Engineering, Section 1B
**Upstream repo:** https://github.com/ascherj/pathreview
**Fork:** https://github.com/Mamadouba2004/pathreview

---

## Week 7 — Issue selection

**Issue link:** https://github.com/ascherj/pathreview/issues/24

**Issue title:** Hybrid retriever over-weights keyword results when query contains technology names

**Tier:** [ ] Tier 1 &nbsp;&nbsp; [x] **Tier 2** &nbsp;&nbsp; [ ] Tier 3

### Problem summary

PathReview retrieves evidence for a review by asking two systems for candidate
chunks — a vector store that scores by semantic similarity and a BM25 keyword
searcher — and then merging the two ranked lists into a single score before the
chunks reach the LLM. The merge in `rag/retriever/hybrid.py` is what's broken:
each side's scores are normalized by dividing by the maximum score *within that
side's own result set*, which pins the single best keyword hit to a normalized
1.0 no matter how weak its absolute BM25 score actually was. So when a query
contains a common technology name like "React" or "Python" — a term that appears
throughout both the resume and every project README — BM25 fires on chunks that
merely mention the word, those chunks receive the artificially inflated score,
and they crowd out the chunks the vector search identified as genuinely relevant;
a chunk found *only* by keyword search carries `vector_score = 0.0` yet can still
clear the `min_score = 0.3` cutoff on keyword weight alone, so text from the
wrong document reaches the generator. The user-visible symptom is portfolio
feedback that cites the wrong project or quotes boilerplate instead of the work
being reviewed. A successful fix puts the two score families on a genuinely
comparable scale (rank-based fusion, or a normalization that doesn't force the
top hit to 1.0), stops rewarding a chunk for the other retriever's silence, and
ships a regression test where a technology-name query provably ranks the relevant
chunk above the keyword-noise chunk.

### Selection notes — "Is this issue right for me?"

**Do I understand what the issue is actually asking?**
Yes. Score fusion between dense and sparse retrieval is the same problem I worked
through building a RAG system over ~14.8k course reviews, where citation quality
depended entirely on retrieving from the correct source document. I recognize this
failure mode rather than guessing at it.

**Can I find and read the affected code?**
Yes, and I read it before claiming. `HybridRetriever` lives in
`rag/retriever/hybrid.py`, and a repository-wide search shows the class is not
constructed anywhere else in the tree — so the change is contained to one class
plus new tests. Its collaborators (`VectorStore`, `KeywordSearcher`) are only read
from, never mutated, which keeps the blast radius small.

**Is the scope honestly what it looks like?**
One caveat I want on the record now rather than at PR time: the issue body
describes "a fixed 50/50 weight," but `HybridRetriever.__init__` currently defaults
to `vector_weight=0.7, keyword_weight=0.3`. The weights are therefore not the whole
story — the max-normalization is doing more damage than the ratio is. I would
rather surface that in the PR than quietly re-tune two constants and call the issue
closed. That widens the reasoning, not the diff.

**Why Tier 2 rather than Tier 1 or Tier 3?**
Tier 2 is the honest calibration for where I am. This is my first contribution to a
codebase of this size, so committing to a Tier 3 architectural change over four
weeks *while* learning the project's conventions would be over-reaching. But Tier 1
issues are scoped to a single file or config, and I already have hands-on RAG
retrieval experience to bring here — I'd learn less from one. Tier 2 asks me to
understand how `rag/retriever/` connects to the generation path without asking me to
redesign it, and the maintainer's 4–6 hour estimate fits the Week 8–9 window.

**What would make me drop or re-scope this?**
If reproducing the ranking bug turns out to require ingesting a realistic
multi-document profile that the seeded fixtures don't provide, then the effort
estimate is wrong and I'd reassess in Week 8 — not at PR time.

### Setup

**Branch name:** `fix/24-hybrid-retriever-keyword-overweight`
(per `docs/CONTRIBUTING.md`: `<type>/<issue-number>-<short-description>`)

**Setup confirmation:** [ ] App runs locally at localhost:5173
*Fork created and branch pushed. Local `docker compose up -d` → `make setup` →
`make run` verification pending; will check this off once the dashboard serves.*

**Cohort ledger:** [x] Issue added to cohort ledger (Section 1b, 2026-07-26)

**Issue claimed:** [x] Comment posted on
[ascherj/pathreview#24](https://github.com/ascherj/pathreview/issues/24)

### Planned next steps (Week 8)

1. Reproduce: write a failing test that seeds a resume chunk and an unrelated
   README chunk both containing "React," then assert the current ranking is wrong.
2. Compare candidate fixes — Reciprocal Rank Fusion vs. min-max or z-score
   normalization over the full candidate pool — against the project's existing
   patterns in `rag/`.
3. Write `PLAN.md` documenting the chosen approach and its trade-offs.

---

## Week 8 — Reproduction & solution planning

**Reproduction commit link:**
[`eaf271f` — test(rag): reproduce issue #24 hybrid retriever keyword over-weighting](https://github.com/Mamadouba2004/pathreview/commit/eaf271f)

**Reproduction summary:**

I reproduced the issue by driving the real `HybridRetriever.retrieve()` against
the real `KeywordSearcher` (actual BM25 scoring) and a fake `VectorStore` that
mirrors `VectorStore.query()`'s `1 / (1 + distance)` similarity conversion, over
a corpus of one resume plus one unrelated project README. Querying `"React"`
ranks `create-react-app` boilerplate first at a blended score of **0.9490**,
ahead of the resume line that actually describes React work at **0.8636** —
confirming the issue is real and locating it in the blending block of
`rag/retriever/hybrid.py`.

The measurement also corrected my Week 7 assumption about the cause. It isn't the
weight ratio. Vector similarities cluster in **0.4082 – 0.5618**, so dividing by
their own maximum leaves the vector signal spanning only ~27% of 0–1, while BM25
starts at 0 and always spans the full range. Multiplying each family's usable
spread by its weight gives vector `0.7 × 0.27 ≈ 0.191` against keyword
`0.3 × 1.00 = 0.300` — **the keyword term controls ~61% of the ranking at a
nominal 30% weight.** The per-retriever max-normalization is the bug, not the
constants.

The new test file `tests/unit/test_hybrid_retriever.py` holds three tests. Two
pass today and pin down the mechanism (the differing dynamic ranges, and a
keyword-only chunk clearing `min_score` on keyword weight alone). The third,
`test_technology_name_query_ranks_relevant_chunk_first`, fails by design and is
the red half of the cycle — it should go green in Week 9 without weakening its
assertion.

**PLAN.md link:**
[PLAN.md on the working branch](https://github.com/Mamadouba2004/pathreview/blob/fix/24-hybrid-retriever-keyword-overweight/PLAN.md)
(added in [`b8a47a0`](https://github.com/Mamadouba2004/pathreview/commit/b8a47a0))

**Walkthrough video (recommended):** not recorded.

**Blockers or open questions:**

1. **`_get_all_chunks()` appears to be dead code, and I don't understand why
   yet.** In `retrieve()` its return value is assigned to `all_chunks` and then
   never used, and `self.keyword_searcher.search()` is called without the
   searcher being indexed anywhere in that method. `KeywordSearcher.search()`
   returns `[]` on an unindexed instance, so either `index()` is called from
   somewhere in the ingestion path I haven't traced, or keyword search silently
   contributes nothing in production — in which case the production symptom
   differs from my reproduction. Tracing the callers of
   `KeywordSearcher.index()` is my first Week 9 task, and it could change the
   fix.
2. **Whether Reciprocal Rank Fusion is the right call for this project.** RRF
   discards score magnitude by design. If anything downstream reads
   `vector_score` / `keyword_score` as raw similarities, rank-derived values
   would be misleading. I've searched for the class name but not yet for those
   dict keys across `api/`, `rag/`, and `frontend/src/`.
3. **No local `make run` verification yet.** The reproduction is a unit-level
   harness, not the full stack, so I have not yet observed the wrong chunk
   surfacing in an actual generated review.

---

## Week 9 — Solution building & PR submission

### Check-in 1 (mid-week)

**Current progress:**

Resolved the first Week 8 blocker before writing any fix code. Traced
`KeywordSearcher.index()` and `HybridRetriever(` across the whole repo (not
just the files I'm touching): zero production callers of either. Reading
`core/services/review_service.py` confirmed why —
`_run_rag_retrieval_generation()` is an explicit placeholder ("Placeholder:
actual RAG logic"), so the RAG retrieval step isn't wired into the live
review pipeline yet in this snapshot of the codebase. `HybridRetriever` is
real, tested-in-isolation code, just not load-bearing in production yet.
That changes the framing of the fix (correctness bug in an unused-but-real
module) without changing the fix itself.

Also ran `make test-unit` and `make check` equivalents before touching
anything, per the Week 9 instructions on pre-existing failures: 39
pre-existing unit test failures and 182 pre-existing ruff errors repo-wide,
none in the fusion logic I'm about to change (one unrelated pre-existing
failure in `test_keyword_search.py::test_empty_index`, a `ZeroDivisionError`
in `rank_bm25` on an empty corpus - not something `#24` touches).

PLAN.md sub-tasks 1 (extract blending into a testable unit) and 2 (replace
per-retriever max-normalization with Reciprocal Rank Fusion) are done -
implemented together rather than as two separate diffs, since a bare
extraction with no behavior change would have been immediately reverted by
step 2 anyway.

**Next steps:**

Finish PLAN.md sub-tasks 3-5: confirm `min_score` still filters correctly
against the rescaled fused scores, add regression coverage for the edge
cases named in PLAN.md, turn the reproduction test green, then open the PR.

**Blockers:**

None blocking. One environment limitation: `chromadb` would not finish
installing in the verification sandbox (large ML dependency, install timed
out) so `vector_store.py` itself isn't exercised - `rag/retriever/hybrid.py`
and `rag/retriever/keyword_search.py` are verified against their real,
unmodified source, with `VectorStore` exercised through the same
`FakeVectorStore` the test file already used before this branch existed.
`vector_store.py` was not modified.

---

### Check-in 2 (end of week)

**PR link:** [#1 — fix(rag): replace per-retriever score normalization with RRF (#24)](https://github.com/Mamadouba2004/pathreview/pull/1)

**Branch:** `fix/24-hybrid-retriever-keyword-overweight`

**What you built:**

Replaced `HybridRetriever.retrieve()`'s per-retriever max-normalization with
weighted Reciprocal Rank Fusion (RRF, k=60). The old code normalized vector
and BM25 scores independently before summing them; because vector
similarities cluster in a narrow band and BM25 scores don't, the nominal
0.7/0.3 weighting was not the effective weighting and keyword noise could
outrank the semantically relevant chunk. RRF fuses on rank position instead
of raw score magnitude, so it's unaffected by the two retrievers having
different score distributions. Also removed `_get_all_chunks()`, which was
dead code (see Check-in 1).

**Tests added or updated:**

`tests/unit/test_hybrid_retriever.py` - the reproduction test from Week 8
(`test_technology_name_query_ranks_relevant_chunk_first`) now passes rather
than failing by design. Added five more: one asserting `score ==
vector_score + keyword_score` (a correctness property the old code didn't
have), and four covering the edge cases named in PLAN.md - a chunk found by
only one retriever, both retrievers returning nothing, a single-candidate
batch, and every BM25 score tying (which surfaces a real, documented
limitation: RRF is blind to score magnitude, not just the old bug).

**Self-review confirmation:** [x] make check passes [x] make test-unit passes
*(scoped to the touched files - see Notes for Reviewers on the PR for exactly
what "passes" verifies in this sandbox, including the two files that could
not be checked at all due to environment gaps.)*

**Draft PR feedback received from:** none

---

## Week 10 — Iteration & reflection

### Reviewer feedback

**Feedback received:** [ ] Yes &nbsp;&nbsp; [x] No — still awaiting review

**Summary of feedback:**
No reviewer feedback came in on [PR #1](https://github.com/Mamadouba2004/pathreview/pull/1). Per the Su26 course note, reviewer feedback isn't a live feature this term, so I'm not expecting a maintainer pass — I'm treating the self-review documented in the PR's "Notes for Reviewers" section as the review substitute for this cycle.

**How you responded:**
N/A — nothing to respond to. If this were a live repo, the three points I flagged for a reviewer myself (RRF's magnitude-blindness, the unwired `HybridRetriever`, the `_get_all_chunks()` deletion) are exactly where I'd expect pushback, and I already wrote out my reasoning for each rather than waiting to be asked.

---

### Reflection

**What was harder than you expected?**
Staying honest about verification under real environment gaps. My sandbox couldn't install `chromadb`, didn't have Python 3.11 (the repo's pinned version), and had no Docker/Postgres — so `make check` and `make test-unit` as literally specified were never runnable end to end. The harder discipline wasn't writing the fix, it was resisting the temptation to write "tests pass" and let that imply more than it should. I ended up writing a "how I verified, and the limits of my sandbox" section in the PR instead, which took longer than just claiming a clean run, but it's the only version that's actually true. The other thing that surprised me: my own Week 7 root-cause theory ("it's the weight ratio") turned out to be wrong once I actually measured it in Week 8. The issue's own description said "a fixed 50/50 weight," and I nearly built the fix around correcting that instead of the real bug, which was the per-retriever normalization.

**What did you learn about working in a large codebase?**
That you have to verify a change is safe before you make it, not after. Before I touched `HybridRetriever.__init__`'s signature I grepped the whole repo for callers — zero outside tests — which is also how I found that `_get_all_chunks()` was dead code and that the class isn't wired into `core/services/review_service.py` at all (`_run_rag_retrieval_generation()` is still a placeholder). That last fact changed how I framed the whole PR: this is a real, measurable bug, but not one currently touching live traffic, and saying that plainly is more useful to a reviewer than either overstating the urgency or hiding the caveat. In my own projects I've never had to reason about "who else calls this" before changing a function — here it was the first thing I did on every change.

**How did AI tools help — and where did they fall short?**
AI was strongest as a pairing partner for structure I already knew the shape of: turning "I need edge cases" into the actual matrix (one retriever silent, both silent, all-tied BM25, single candidate) faster than I'd have enumerated them alone, and helping draft the PR template sections so I could focus on getting the content right rather than the formatting. It fell short anywhere that required judgment I had to own: deciding RRF's magnitude-blindness was an acceptable trade-off for this codebase (not a fact to look up — a call to make and defend), and catching a subtle framing problem myself where an early draft of one test's expected score numerically resembled the old bug's forced output in a way that could have implied the fix changed nothing, when it hadn't actually failed — I had to notice that and reframe the assertion around the real invariant instead of a coincidental number. AI also can't run `chromadb` in an environment that won't install it; it can help you document that gap honestly, but it can't paper over it.

**What would you do differently if you started over?**
I'd hit the Python 3.11 / `chromadb` requirement in Week 7 setup instead of discovering the gap while trying to verify in Week 9 under deadline pressure — that would have given me time to either fix the environment or plan the lightweight-harness verification strategy from the start instead of assembling it last-minute. I'd also measure the score distributions (vector vs. BM25 dynamic range) in Week 7 rather than Week 8, since that's what actually overturned my root-cause assumption — catching it a week earlier would have meant PLAN.md was built on the right diagnosis from day one instead of course-correcting mid-flight. And I'd open the PR earlier in Week 9 rather than at the very end, per the course's own advice not to wait for the due date — I did most of the drafting right up against the deadline, which left no real window for the review cycle this week is nominally about.

**What are you most proud of from this module?**
Not the fix itself — the decision to put the "this bug is real but not currently live-traffic-affecting" caveat directly in the PR, right next to the fix, instead of either burying it or leaving it out to make the contribution look more consequential than it is. That's the kind of claim that's easy to shade in either direction, and I think getting it exactly right — measured, not oversold, not underclaimed — is closer to what a real maintainer would actually want from a contributor than a technically-correct fix with an inflated pitch around it.
