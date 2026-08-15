# Eval Harness Spec — agentic-RAG

Goal: make tuning the RAG pipeline *measurable*. Today the harness reports
numbers, but they are not trustworthy (see §2). This spec defines the metrics,
the golden dataset shape, the harness architecture, and acceptance thresholds.
NO code is written yet — this is the alignment doc.

## 1. What "effective" means here
A RAG answer is effective if:
- (Retrieval) the right passage(s) were pulled and ranked high, and
- (Generation) the answer is *grounded in* the retrieved context (not made up),
  actually answers the question, and is factually correct vs the expected answer,
- (Safety) questions with no answer in the docs are refused, not hallucinated.

## 2. Current deficiencies (why the numbers lie)
- `Recall@k` is **title-binary**: `hits = [c for c in chunks if c.title == source]`.
  Corpus is 2 docs; at top_k=4 both docs appear, so recall is trivially 1.0.
- `MRR` is computed over doc-title hits, not passages — meaningless on 2 docs.
- **No faithfulness/groundedness**: the judge compares answer vs expected string
  only; it never checks the answer is supported by *retrieved context*. Hallucination is invisible.
- **No retrieval precision / NDCG**: cannot tell if tuning top_k / candidate_k /
  similarity_threshold actually improves ranking.
- **Corpus has no distractors, no cross-doc or multi-hop questions, no passage-level goldens.**
  28 questions, all "answerable" ones map to one doc title.

## 3. Metric taxonomy

### Retrieval (no LLM needed — deterministic)
Computed from retrieved chunks vs golden passage spans.
| Metric | Definition | Why |
|---|---|---|
| Context Recall | fraction of golden answer-passages present in retrieved set (top_k) | "Did we fetch what's needed?" |
| Context Precision@k | of top-k chunks, fraction that are golden-relevant, by rank | "Is relevant stuff on top?" |
| MRR@k | mean reciprocal rank of first golden passage | ranking quality |
| NDCG@k | graded relevance of top-k (golden passage weight) | ranking quality, handles partial |
| Hit Rate@k | 1 if any golden passage in top-k else 0 | coarse retrieval success |

### Generation (LLM-as-judge, grounded)
| Metric | Definition | Why |
|---|---|---|
| Faithfulness / Groundedness | are the answer's claims supported by retrieved context? | **the missing anti-hallucination check** |
| Answer Relevancy | does the answer address the question? | stops off-topic answers |
| Answer Correctness | factual overlap with expected answer (LLM judge, upgraded from current string match) | quality of the answer |
| Not-found rate (unanswerables) | fraction of unanswerable Qs correctly refused | safety, already partly present |

## 4. Golden dataset schema (replaces current golden.jsonl)
Each line (JSONL):
```
{
  "question": "...",
  "unanswerable": false,
  "expected": "The SunPak 5 stores 5.1 kWh of usable energy.",
  "golden_passages": ["<exact substring or chunk id that answers it>"],
  "golden_doc": "helios_energy_handbook.md",   // for cheap title check (kept)
  "needs": ["single_passage" | "multi_passage" | "multi_doc" | "multi_hop"],
  "type": ["fact" | "procedure" | "definition" | "negation"]
}
```
- `golden_passages` are exact substrings copied from the corpus so recall/precision
  are computed by substring/embedding match against retrieved chunk text.
- `needs` drives coverage: we want at least a few `multi_doc` and `multi_hop` cases
  so the harness can't score 1.0 by accident.

## 5. Corpus requirements (to make metrics meaningful)
- Expand from 2 → **6–8 documents** across 2–3 domains (keep Helios + Meridian, add e.g.
  a policy doc, a product manual, a FAQ) so distractors exist.
- Add **distractor passages** (topically near but not the answer) inside docs.
- Questions must include: single-passage (most), multi-passage, **multi-doc**
  (answer spans 2 docs), **multi-hop** (answer requires combining 2 facts), and
  unanswerable (negative) cases.
- Target ~40 questions (vs current 28), with ≥4 multi-doc and ≥4 multi-hop.

## 6. Harness architecture (run_eval refactor)
- `retrieve()` already returns chunks with `text` + `similarity` + `title`.
  Keep it; add `chunk_id` to support exact golden-passage matching.
- New `metrics.py` (eval/metrics.py): pure functions for Context Recall,
  Precision@k, MRR@k, NDCG@k, Hit Rate@k — take retrieved chunks + golden_passages.
- New judges in `run_eval.py` / `judges.py`:
  - `faithfulness(question, context, answer)` — strict "is every claim in context?"
  - `answer_relevancy(question, answer)`
  - `answer_correctness(question, expected, answer)` — upgrade of current judge
- `golden_passages` matched by **embedding-cosine** against retrieved chunk text,
  using the SAME embedding model as the pipeline (consistency, no new machinery).
  RESOLVED (2026-08-15): user chose to keep embedding-cosine matching as-is.
  Measurement-first — revisit matching only if Context Recall numbers prove it's wrong.
- Report (`eval/runs/<ts>/report.json` + `.md`) gains a `retrieval` and `generation`
  block; config snapshot unchanged (already good).
- `compare.py`: extend METRICS list to the new keys; show deltas per metric.

## 7. Acceptance thresholds (targets, not current values)
| Metric | Target |
|---|---|
| Context Recall | ≥ 0.80 |
| Context Precision@k | ≥ 0.70 |
| MRR@k | ≥ 0.65 |
| NDCG@k | ≥ 0.70 |
| Faithfulness | ≥ 0.90 |
| Answer Relevancy | ≥ 0.85 |
| Answer Correctness | ≥ 0.80 |
| Not-found rate (unanswerables) | ≥ 0.90 |
A tuning change counts as an improvement only if retrieval + generation rise
**without** the not-found rate dropping (PRD §7.4 rule, already stated).

## 8. Implementation phases (no code yet)
1. Expand corpus to 6–8 docs + distractors; rewrite golden.jsonl with passage-level goldens + needs/type tags.
2. Add `chunk_id` to retrieved chunks; write `eval/metrics.py` (retrieval metrics).
3. Write `eval/judges.py` (faithfulness, relevancy, correctness).
4. Refactor `run_eval.py` to emit both retrieval + generation metric blocks.
5. Extend `compare.py` METRICS.
6. Add `--retrieval-only` and full modes; keep config snapshot.

## 9. Verification plan
- Deterministic stub run: a tiny 2-doc corpus with known goldens where retrieval
  is intentionally weakened (top_k=1, threshold high) should drop Context Recall
  below target — proving the metric reacts, not stuck at 1.0.
- Full run on expanded corpus: all targets listed; a "bad config" (e.g.
  similarity_threshold=0.9) must visibly degrade retrieval metrics.
- `compare --last-two` shows signed deltas for every new metric.

## 10. Resolved user decisions (2026-08-15)
- **Chat scope:** keep GLOBAL per-user chat. Do NOT adopt ON's notebook-scoped
  sessions. Multi-corpus = more sources in the same user's space; per-user
  isolation already handled by per-(user, embedding_model) Chroma collections.
- **Judge model:** reuse the same proxy LLM (qwen3.8-max) for faithfulness /
  relevancy / correctness judges.
- **Golden-passage matching:** embedding-cosine, same model as pipeline. Keep as-is.
  Measurement-first — improve matching only if metrics demand it.

## 11. Open questions (remaining)
- Corpus: expand inline with synthetic docs I author (keep Helios/Meridian + add
  ~4 more domains with distractors), or point me at real documents?
