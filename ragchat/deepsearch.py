"""Deep search: an exhaustive literal scan of the user's own documents.

Retrieval ranks. This does not.

Vector search and BM25 both answer "which chunks look most like this question",
and both operate on CHUNKS — so a phrase that exists verbatim in a document can
still be absent from the answer, because the chunk holding it placed 21st in a
pool of 20. That failure is invisible: the model answers from what it was given
and has no way to know what it was not. It is the single most common reason a
grounded system says something is not in a document that plainly contains it.

Deep search closes that gap by not competing for a place. It reads the full
source text of every document the user owns, finds every literal occurrence of
the question's terms, and returns the surrounding windows. If the words are
there, they are found — corpus size changes how long it takes, never whether it
succeeds.

Three deliberate properties:

- It searches `Document.source_text`, which the sliced-ingest path already
  stores durably (db.py). No new storage, no embedding call, no vector-backend
  dispatch, and nothing that can drift out of step with the index.

- Its passages carry `similarity: None`, exactly like BM25-only chunks. They
  therefore take no part in the "do the user's documents answer this?" decision,
  and the reranker scores them against the question alongside everything else.
  A literal match is a candidate, not a verdict: "the" appears in every
  document, and finding it means nothing.

- It is a PER-REQUEST choice, never a stored setting. `config_overrides` is a
  single row shared by the whole deployment (db.py), so a persisted toggle here
  would change retrieval for every user at once — the bug the web-augmentation
  toggle it replaces actually had.
"""
from __future__ import annotations

import re
import unicodedata

from sqlalchemy.orm import Session

from .db import Document

# Words too common to be evidence of anything. Deliberately short: this is not
# linguistics, it is a guard against a window scoring highly because it contains
# "what" and "the". Anything domain-specific stays in, because in a document
# corpus the domain words are exactly what the visitor is looking for.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with about into over after before between out against during is are was were
be been being do does did doing have has had having can could should would may
might will shall must i you he she it we they what which who whom whose when
where why how not no nor so such as any all each both few more most other some
""".split())

_TOKEN = re.compile(r"[\w']+")
_CJK_CHAR = r"[㐀-䶿一-鿿豈-﫿]"
_CJK = re.compile(_CJK_CHAR)
# A whole run of CJK. Python's \w MATCHES CJK, so a plain tokenizer turns
# 周五数据不更新会怎样 into one "word" that appears verbatim in no document,
# and deep search silently finds nothing at all — which reads as "not in the
# corpus" rather than as a bug. Runs are handled separately below.
_CJK_RUN = re.compile(_CJK_CHAR + "+")

# Text either side of a hit. Wide enough that a matched figure arrives with the
# sentence that gives it meaning — "S$9,800" alone tells the model nothing about
# what it is the price of.
WINDOW_CHARS = 420

# Caps. Deep search runs inside one serverless request beside a rerank and a
# generation call, so its cost has to be bounded by construction rather than by
# hoping corpora stay small.
MAX_PASSAGES = 8
MAX_HITS_PER_TERM = 40
MAX_DOC_CHARS = 400_000


def _terms(query: str) -> list[str]:
    """Search terms, longest first, quoted phrases kept whole."""
    terms: list[str] = []
    rest = query
    # "exact phrase" wins: someone who quotes it is telling us the words belong
    # together, and a phrase is far stronger evidence than its parts.
    for phrase in re.findall(r'"([^"]{2,80})"', query):
        terms.append(phrase.strip())
        rest = rest.replace(f'"{phrase}"', " ")

    # CJK first, and removed from the text before word tokenizing, because \w
    # would otherwise consume the whole run.
    #
    # Terms are character BIGRAMS, not whole runs and not single characters.
    # A run is a clause and rarely appears verbatim anywhere; a single character
    # is 的 or 不 and appears everywhere. Bigrams are the standard
    # segmenter-free middle: 周五数据 gives 周五 / 五数 / 数据, of which two are
    # real words and the third is harmless — merged windows collapse them into
    # one passage anyway.
    for run in _CJK_RUN.findall(rest):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    rest = _CJK_RUN.sub(" ", rest)

    for tok in _TOKEN.findall(rest):
        tok = tok.strip("'")
        if len(tok) < 3 or tok.lower() in _STOPWORDS:
            continue
        terms.append(tok)

    # Longest first so a window is attributed to the most specific term it
    # contains, and de-duplicated case-insensitively.
    seen: set[str] = set()
    out: list[str] = []
    for t in sorted(terms, key=len, reverse=True):
        k = t.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _fold(text: str) -> str:
    """Normalise and casefold, so a query matches regardless of case or form.

    Offsets from the folded copy are used to slice the ORIGINAL, so the two must
    stay the same length. NFKC usually keeps it — but not always (ligatures and
    full-width forms expand), and a length change would shift every window by a
    character or more. literal_passages() checks the length and falls back to a
    plain casefold when it differs, which is length-stable for the scripts here.
    """
    return unicodedata.normalize("NFKC", text).casefold()


def _windows(text: str, folded: str, terms: list[str]) -> list[tuple[int, int, set[str]]]:
    """Merged (start, end, matched-terms) spans around every literal hit."""
    spans: list[tuple[int, int, str]] = []
    for term in terms:
        needle = _fold(term)
        if not needle:
            continue
        start = 0
        found = 0
        while found < MAX_HITS_PER_TERM:
            i = folded.find(needle, start)
            if i < 0:
                break
            spans.append((max(0, i - WINDOW_CHARS),
                          min(len(text), i + len(needle) + WINDOW_CHARS),
                          term))
            start = i + len(needle)
            found += 1
    if not spans:
        return []

    spans.sort()
    merged: list[tuple[int, int, set[str]]] = []
    for lo, hi, term in spans:
        if merged and lo <= merged[-1][1]:
            p_lo, p_hi, p_terms = merged[-1]
            merged[-1] = (p_lo, max(p_hi, hi), p_terms | {term})
        else:
            merged.append((lo, hi, {term}))
    return merged


def literal_passages(
    db: Session,
    user_id: str,
    query: str,
    *,
    max_passages: int = MAX_PASSAGES,
) -> list[dict]:
    """Every literal hit for `query` in this user's documents, best windows first.

    Returns chunk-shaped dicts so the rest of the pipeline treats them like any
    other passage.
    """
    terms = _terms(query)
    if not terms:
        return []

    rows = (
        db.query(Document.id, Document.title, Document.source_text)
        .filter(
            Document.user_id == user_id,
            Document.source_text.isnot(None),
        )
        .all()
    )

    scored: list[tuple[int, int, dict]] = []
    for doc_id, title, source_text in rows:
        if not source_text:
            continue
        text = source_text[:MAX_DOC_CHARS]
        folded = _fold(text)
        # NFKC can change length (ligatures, full-width forms), which would
        # shift every offset. When it does, fall back to a plain casefold,
        # which is length-stable for the scripts this corpus uses.
        if len(folded) != len(text):
            folded = text.casefold()
        for lo, hi, matched in _windows(text, folded, terms):
            excerpt = text[lo:hi].strip()
            if not excerpt:
                continue
            # Distinct terms first, then total length matched. A window holding
            # three of the question's words beats one holding the same word
            # three times — which is what a raw count would have preferred.
            scored.append((
                len(matched),
                sum(len(t) for t in matched),
                {
                    "text": excerpt,
                    # None, like a BM25-only chunk: this is a literal hit, not a
                    # measured semantic distance, and reporting a number here
                    # would let it vote in the not-found decision it has no
                    # business in. pipeline._fallback_score handles None.
                    "similarity": None,
                    "doc_id": doc_id,
                    "title": title,
                    "ref": f"~{round(lo * 100 / max(len(text), 1))}% of document",
                    "deep": True,
                },
            ))

    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [d for _, _, d in scored[:max_passages]]


def searcher(db: Session, user_id: str):
    """Bind a session and a user, so the pipeline never imports the ORM.

    The pipeline takes a callable of the REWRITTEN query — deep search should
    scan for the terms the retrieval actually used, not the raw question, or a
    follow-up like "and the second one?" would search for the word "second".
    """
    def _search(query: str) -> list[dict]:
        return literal_passages(db, user_id, query)

    return _search
