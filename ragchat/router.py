"""Which tool should the app reach for? Asked of a model, not of an `if`.

The escalation already knows WHEN to look harder — retrieval fell below the
threshold, or the model read the passages and said the answer was not there.
What it could not do is decide WHICH tool suits the question, because the model
that writes the answers does not emit tool calls at all.

Verified live against the endpoint, same schema and prompt:

    models/gemma-4-26b-a4b-it     accepts `tools`, never calls one — it
                                  reasons about the choice in prose instead
    models/gemini-3.5-flash-lite  emits a proper tool call in ~0.7s

So the two jobs are split. Gemma keeps writing every answer, because it scores
6.5 points higher on answer correctness than flash-lite over the golden set and
that is the number the reader feels. Flash-lite only ever picks a tool; nothing
it produces is shown to anyone.

The distinction it can make and a rule cannot: "the answer should be in these
documents, the ranker just missed it" versus "this could never have been in a
private document at all". The first is a job for the literal scan, the second
for the web, and telling them apart is a judgement about the question rather
than a threshold on a score.

FAILS OPEN, always. A router that errors, times out, or declines to choose
returns None, and `ask()` falls back to the fixed order it used before this
module existed. The router can improve the choice; it can never block it.
"""
from __future__ import annotations

import logging

from .config import PipelineConfig
from .embeddings import openai_client, retry_call

log = logging.getLogger(__name__)

# Kept in the same words the tools describe themselves in, because this text is
# the entire basis on which the model chooses. "Use ONLY when" on the web tool
# is doing real work: without it the model reaches outside for anything it
# cannot immediately see, which is the behaviour that got web search deleted
# the first time.
_TOOL_SPECS: dict[str, dict] = {
    "deep": {
        "type": "function",
        "function": {
            "name": "scan_documents_literally",
            "description": (
                "Read every one of the user's own documents word for word and "
                "return literal matches. Use when the answer is plausibly IN "
                "their documents but ranked search may have missed the exact "
                "wording — part numbers, form codes, names, rare phrases."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    "web": {
        "type": "function",
        "function": {
            "name": "search_the_web",
            "description": (
                "Search the public web. Use ONLY when the answer could not "
                "plausibly be in the user's own private documents at all — "
                "current events, public reference facts, third-party products "
                "they do not own documentation for."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
}

_NAME_TO_TOOL = {spec["function"]["name"]: key for key, spec in _TOOL_SPECS.items()}

SYSTEM = (
    "You choose the next action for a question-answering app that answers from "
    "the user's own documents. You are given the question and the passages "
    "retrieval returned, which were judged insufficient. Choose exactly one "
    "tool, or none if no tool would help. Always prefer the user's own "
    "documents over the web."
)

# The passages are context for a judgement, not material for an answer, so they
# are trimmed hard. A router that reads 8000 tokens to pick between two options
# costs more than the retry it is choosing for.
MAX_PASSAGES = 4
MAX_PASSAGE_CHARS = 300


def _describe(passages: list[dict]) -> str:
    if not passages:
        return "(retrieval returned nothing at all)"
    lines = []
    for c in passages[:MAX_PASSAGES]:
        text = " ".join((c.get("text") or "").split())[:MAX_PASSAGE_CHARS]
        lines.append(f"- {c.get('title') or 'untitled'}: {text}")
    return "\n".join(lines)


def choose_tool(
    query: str,
    passages: list[dict],
    available: list[str],
    cfg: PipelineConfig,
) -> str | None:
    """Pick one of `available` ('deep' / 'web'), or None for no preference.

    None is not a failure signal — it is "no opinion", and the caller treats it
    the same whether it means the router declined, broke, or was never
    configured.
    """
    tools = [_TOOL_SPECS[name] for name in available if name in _TOOL_SPECS]
    if len(tools) < 2:
        # One tool is not a choice. Skipping the call here is what keeps the
        # router from costing anything on a deployment where web search is
        # unconfigured or the visitor switched a tool off.
        return None

    prompt = (
        f"Question: {query}\n\n"
        f"Passages retrieved from the user's documents:\n{_describe(passages)}"
    )
    try:
        resp = retry_call(
            openai_client().chat.completions.create,
            model=cfg.router_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            tool_choice="auto",
            # Enough for a tool call and nothing else. The router must not be
            # able to spend a paragraph deciding.
            max_tokens=200,
            temperature=0.0,
        )
        calls = getattr(resp.choices[0].message, "tool_calls", None) or []
        if not calls:
            return None
        return _NAME_TO_TOOL.get(calls[0].function.name)
    except Exception:
        # Never fatal. `ask()` falls back to the fixed order, which is exactly
        # what it did before this module existed.
        log.exception("tool router failed; falling back to the fixed order")
        return None
