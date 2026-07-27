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
