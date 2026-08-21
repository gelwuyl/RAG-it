"""Web search as a TOOL the app may reach for — not a fallback it falls into.

Web augmentation existed here once and was deleted, for two reasons worth
keeping in front of whoever changes this file:

  * it answered "your documents did not have it" by looking somewhere else
    instead of by looking harder, which is the opposite of what a
    document-grounded app is for; and
  * it lived in `config_overrides`, a single row shared by the whole
    deployment, so one visitor flipping "their" switch changed retrieval for
    everyone.

It is back under different rules. It is the LAST rung of the escalation ladder
in `pipeline.ask` — reached only after the user's own documents have been
searched twice, by ranking and then literally — and it rides on the request, so
nothing about it is ever written to shared config.

The passages it returns are the only ones in the pool that are not from the
user's own documents. That distinction is the app's whole promise, so it is
carried explicitly (`web: True`), labelled in the prompt, and badged in the
citation. Losing that flag anywhere along the way turns "here is what the web
says" into "here is what your documents say", which is the one failure this
feature must never have.

Provider is Tavily: a search API that returns SNIPPETS, which is the shape the
pool already expects. An LLM-with-browsing endpoint would return a finished
answer instead, and an answer cannot be reranked against the user's documents
or cited passage by passage.
"""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("ragchat.websearch")

ENDPOINT = "https://api.tavily.com/search"

# How many results reach the pool. Deliberately small: they are competing for
# context against the user's own documents, and the reranker orders the merged
# pool anyway, so a wide net here mostly buys tokens.
MAX_RESULTS = 4

# A snippet longer than this is a page, not a passage. Matches the order of
# magnitude of deepsearch.WINDOW_CHARS so one source cannot dominate the
# context purely by being verbose.
MAX_CHARS = 600

# The escalation has already spent one generation by the time this runs, and
# the whole request has 60 seconds. A slow search must fail rather than eat the
# budget that the retry needs.
TIMEOUT_SECONDS = 10.0


def api_key() -> str:
    """Read at CALL time, never at import.

    A module-level constant would freeze whatever the environment held when the
    process booted — the same mistake that pinned `judges.JUDGE_MODEL` to a
    stale default and made every judge call 404 (CLAUDE.md).
    """
    return os.environ.get("TAVILY_API_KEY", "")


def is_configured() -> bool:
    return bool(api_key())


def _clean(text: str) -> str:
    return " ".join((text or "").split())[:MAX_CHARS]


def web_passages(query: str, *, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search the web and return chunk-shaped passages.

    Returns [] rather than raising on any failure — an unreachable search
    provider must cost the reader nothing, because by the time this is called
    there is already an answer (a refusal) in hand.
    """
    key = api_key()
    if not key or not (query or "").strip():
        return []

    try:
        r = httpx.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "query": query,
                "max_results": max_results,
                # "basic" is one credit; "advanced" is two and returns longer
                # extracts. The reranker is what decides what actually reaches
                # the answer, so paying double for more text to discard is not
                # a good trade on a free tier.
                "search_depth": "basic",
                # The provider will happily write the answer itself. We do not
                # want it: this app's answer must come from its own model, over
                # a pool that also contains the user's documents.
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception:
        log.exception("web search failed for query %r", query[:80])
        return []

    out: list[dict] = []
    for item in (payload.get("results") or [])[:max_results]:
        text = _clean(item.get("content") or item.get("raw_content") or "")
        url = item.get("url") or ""
        if not text or not url:
            continue
        out.append(
            {
                "text": text,
                # No measured distance, exactly like a deep-search hit. The
                # not-found guard reads only passages that HAVE a cosine, so a
                # web result never votes in a judgement it has no standing in
                # — and pipeline._fallback_score is written for this None.
                "similarity": None,
                # Deliberately None: this is not one of the user's documents,
                # and giving it a doc_id would let it be mistaken for one by
                # anything that looks up sources by id.
                "doc_id": None,
                "title": (item.get("title") or url)[:200],
                "ref": url,
                "web": True,
            }
        )
    return out


def searcher():
    """Bind nothing; return the tool in the same shape as deepsearch.searcher.

    Both tools are callables of the REWRITTEN query returning chunk-shaped
    dicts, so `ask()` can hold them in the same hand and the loop that chooses
    between them needs no special case for either.
    """
    def _search(query: str) -> list[dict]:
        return web_passages(query)

    return _search


if __name__ == "__main__":  # pragma: no cover
    # Verify a key end to end without booting the app:
    #   .venv/Scripts/python -m ragchat.websearch "tesla powerwall warranty"
    import sys

    logging.basicConfig(level=logging.INFO)
    if not is_configured():
        print("TAVILY_API_KEY is not set — the tool is disabled, not broken.")
        raise SystemExit(1)
    q = " ".join(sys.argv[1:]) or "what is retrieval augmented generation"
    hits = web_passages(q)
    print(f"{len(hits)} passage(s) for {q!r}\n")
    for h in hits:
        print(f"- {h['title']}\n  {h['ref']}\n  {h['text'][:160]}...\n")
