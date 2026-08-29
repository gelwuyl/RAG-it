"""LLM-as-judge metrics for answer and context quality.

The reference-based correctness judge is used by the benchmark. The live chat
grades the FOUR canonical RAGAS metrics — faithfulness and answer relevancy for
the generation, rank-aware context precision and reference-based context recall
for the retrieval — plus synthesize_expected, which DRAFTS a reference from the
retrieved passages when no human one exists (a matched demo-bank question's
HUMAN answer outranks any draft). Drafted references are estimates, not
golden-set recall; callers label them estimated in the UI. The retired live
proxies (context relevance, context sufficiency, answer correctness) are kept
below because the benchmark CLI harness still calls them.

All judges reuse the same proxy LLM (qwen3.8-max) per user decision 2026-08-15.
The judge returns PASS/FAIL (or a 0-1 score) plus one line of reasoning.
The eval harness calls the reference-based judges only in the full
(non --retrieval-only) run.

Tightened 2026-08-16: the judge prompts now forbid chain-of-thought and force a
single-line verdict + one-sentence reason. Previously the model could emit a
thinking trace before the verdict, which leaked into the parsed "reason" and
occasionally flipped the verdict parse. We also cap max_tokens so the model
cannot ramble, and the parser only reads the verdict line + first sentence.

Tightened 2026-08-29 (judge-discrimination pass): both faithfulness judges and
the scored relevancy judge stopped trusting the model's own vibe-read. The
faithfulness judges now make the judge MARK each claim SUPPORTED/UNSUPPORTED,
one line per claim, and the ratio/verdict is computed here in Python (the
context_precision pattern). The relevancy and precision prompts bar the
benefit of the doubt and demand specificity — the published run's flat 100%
pass rates on faithfulness and relevancy came from judges that charitably
passed anything on-topic. Fail-open semantics are unchanged: a reply without
a usable verdict, or with marks that do not cover every claim, is a
JudgeError ("not graded"), never a guessed score.
"""
from __future__ import annotations

import re

from ragchat.embeddings import openai_client
from ragchat.config import load_config, settings


class JudgeError(RuntimeError):
    """The judge could not produce a verdict (model 404, quota, empty reply).

    Raised instead of silently returning FAIL so callers can distinguish
    "the answer is bad" from "the grader is broken".
    """


# What synthesize_expected returns as the reason when the passages cannot
# answer the question, and what ragchat.pipeline stores in eval_data's
# expected_reason to mark correctness as graded-by-refusal. Kept here because
# the judge is the producer; the pipeline imports this constant rather than
# duplicating a magic string on both sides.
NO_ANSWER_DERIVABLE_SENTINEL = "no answer derivable"


def judge_model() -> str:
    """Model used for LLM-as-judge — resolved from the LIVE config each call.

    This used to be a module-level constant read from ``settings.default_llm_model``,
    i.e. the ``RAG_LLM_MODEL`` env var at import time. That is the *boot default*,
    not what the app is actually generating with: config.yaml (and the DB
    override written by the Settings UI) can name a different model. On this
    deployment the env default was the bare ``gemma-4-26b-it`` while the served
    id is ``models/gemma-4-26b-a4b-it``, so every judge call 404'd and both
    metrics rendered as FAIL. Reading the live config keeps the judge on whatever
    config actually says.

    Since the dedicated ``judge_model`` config field exists: when it names a
    model, that model grades — grading wants terse consistency and speed, and a
    non-thinking judge cannot burn its token budget reasoning inside <thought>
    (measured: scored readings timed out 5/5 under a thinking judge on full
    documents). When the field is EMPTY, the answerer grades — the historical
    behaviour, and the safe fallback if the configured judge ever 404s is a
    JudgeError (fail-open to "not graded"), never a wrong verdict.
    """
    try:
        live = load_config()
        own = (getattr(live, "judge_model", "") or "").strip()
        if own:
            return own
        # No dedicated judge configured: the answerer grades.
        model = (live.llm_model or "").strip()
        if model:
            return model
    except Exception:
        pass
    return settings.default_llm_model

# Strict output contract the judge must follow. We repeat it in every prompt so
# the parser can rely on "first line = verdict, second line = one sentence".
_VERDICT_CONTRACT = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these two lines:\n"
    "VERDICT: PASS or FAIL\n"
    "REASON: one short sentence citing the specific evidence.\n"
)

# The live scorecard draws per-answer readings on the same percentage bars the
# benchmark reports pass rates on, so a binary verdict can only ever fill a bar
# at 0% or 100% — indistinguishable from a measurement. The scored variants ask
# for a 0-100 reading too. The BENCHMARK keeps using the boolean judges above:
# its published numbers are pass rates over 53 golden questions, and switching
# its judges would silently change what those numbers mean.
#
# "Do NOT show your working" is load-bearing, not decorative: with a longer
# prompt the thinking-capable judge model reasons at length about each passage,
# burns the whole token budget, and the reply dies inside <thought> before any
# verdict — five-for-five on the preview deployment before this line existed.
_VERDICT_CONTRACT_SCORED = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these three lines. Do NOT show your working — "
    "decide silently and output only:\n"
    "VERDICT: PASS or FAIL\n"
    "SCORE: an integer from 0 to 100\n"
    "REASON: one short sentence citing the specific evidence.\n"
)

_SCORE_LINE = re.compile(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

# The context-precision judge's contract: one mark line per passage, in
# retrieval order, then the usual verdict pair. The marks are the measurement
# (average precision is computed from them, not from the judge's own SCORE);
# the verdict line exists so a truncated reply fails the same "no verdict"
# check every other judge fails through.
_MARKS_CONTRACT = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these lines. Do NOT show your working — decide "
    "silently and output only:\n"
    "PASSAGE 1: RELEVANT or IRRELEVANT\n"
    "PASSAGE 2: RELEVANT or IRRELEVANT\n"
    "(exactly one line for EVERY passage, in order)\n"
    "VERDICT: PASS or FAIL\n"
    "REASON: one short sentence citing the specific evidence.\n"
)

# The faithfulness judges' contract, on the context_precision model: the judge
# MARKS each claim and the ratio/verdict is computed in Python. A judge asked
# for one aggregate vibe-read passes nearly everything (the published run's
# flat 100%); a judge asked to commit to a mark per claim has to defend each
# one, and a mark that cannot be defended becomes UNSUPPORTED. The verdict
# line is the fail-open anchor only — a reply without it fails through the
# same "no verdict" path every other judge uses.
_CLAIMS_CONTRACT = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these lines. Do NOT show your working — decide "
    "silently and output only:\n"
    "CLAIM 1: SUPPORTED or UNSUPPORTED\n"
    "CLAIM 2: SUPPORTED or UNSUPPORTED\n"
    "(exactly one line for EVERY distinct factual claim in the answer, "
    "numbered in order)\n"
    "If and only if the answer makes NO factual claims at all (e.g. a proper "
    "'the documents do not say' refusal), output the single line "
    "CLAIM 1: NONE\n"
    "VERDICT: PASS or FAIL\n"
    "REASON: one short sentence citing the specific evidence.\n"
)


# Reasoning-tuned models wrap output in these; strip before parsing so a
# thinking trace can neither hide the verdict nor leak into the reason.
_THINK_BLOCK = re.compile(
    r"<(thought|thinking|reasoning)[\s>].*?</\1>", re.IGNORECASE | re.DOTALL
)

# The SAME wrappers, but never closed — what you get when the reasoning trace
# runs into max_tokens and the reply is cut mid-thought. The configured judge
# (models/gemma-4-26b-a4b-it) is thinking-capable and emits <thought>, so this
# is reachable in production, and it defeated the closed-tag pattern above:
# the raw trace survived to the parser, which found a stray "FAIL" inside the
# reasoning and reported it as a confident verdict. A truncated reply is a
# BROKEN GRADER, not a failed answer — strip the unterminated block so nothing
# is left to parse and the caller records "not graded".
_UNCLOSED_THINK_BLOCK = re.compile(
    r"<(thought|thinking|reasoning)[\s>].*\Z", re.IGNORECASE | re.DOTALL
)


def _judge(prompt: str, max_tokens: int = 512) -> str:
    """Ask the judge model for a verdict. Raises JudgeError if it can't answer.

    ``max_tokens`` is deliberately generous (was 96). On thinking-capable models
    the reasoning tokens are billed against this budget, so a tight cap made the
    model spend the whole allowance thinking and return EMPTY content — which
    the old parser then scored as FAIL. The scored judges pass 4096: even 2048
    was exhausted mid-<thought> on full-document contexts (finish_reason="length").
    """
    client = openai_client()
    model = judge_model()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # 404 on an unserved model, quota, network...
        raise JudgeError(f"judge model {model!r} failed: {exc}") from exc
    choice = resp.choices[0] if resp.choices else None
    out = ((choice.message.content if choice else None) or "").strip()
    if not out:
        finish = getattr(choice, "finish_reason", None) if choice else None
        raise JudgeError(
            f"judge model {model!r} returned empty content (finish_reason={finish})"
        )
    return out


def _parse_verdict(out: str) -> tuple[bool, str]:
    """Parse a strict 'VERDICT: PASS/FAIL\\nREASON: ...' response.

    Robust to minor formatting drift: finds the PASS/FAIL token anywhere, and
    takes the reason from the REASON: line (or the text after the verdict),
    truncated to the first sentence.

    Raises JudgeError when no verdict token is present. It previously returned
    ``(False, "")`` in that case — failing CLOSED — so any unparseable reply was
    indistinguishable from a genuine hallucination finding, and the UI showed a
    confident FAIL with no reason. "Unknown" and "failed" are different states
    and must not be conflated.
    """
    text = _THINK_BLOCK.sub("", out or "")
    # Order matters: closed blocks first, so a well-formed trace followed by a
    # real verdict keeps its verdict. Only a block left open to end-of-string
    # (i.e. a truncated reply) is removed here.
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    if not text:
        raise JudgeError("judge returned only a reasoning block, no verdict")

    # Look for the verdict on its labelled line first; only then anywhere.
    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if verdict_match is None:
        verdict_match = re.search(r"\b(PASS|FAIL)\b", text, re.IGNORECASE)
    if verdict_match is None:
        raise JudgeError(f"no PASS/FAIL verdict in judge reply: {text[:120]!r}")
    verdict = verdict_match.group(1).upper() == "PASS"

    # Prefer an explicit REASON: line; otherwise the text after the verdict line.
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # Strip the verdict token and any leading label, take what remains.
        reason = re.sub(
            r"VERDICT:\s*(PASS|FAIL)", "", text, flags=re.IGNORECASE
        ).strip()
    # First sentence only — drop any trailing ramble.
    reason = re.split(r"(?<=[.!?])\s", reason)[0].strip()
    return verdict, reason


def _parse_verdict_scored(out: str) -> tuple[bool, float, str]:
    """Parse VERDICT/SCORE/REASON. Returns (verdict, score 0.0-1.0, reason).

    Raises JudgeError when there is no verdict — the same fail-open rule as
    _parse_verdict. A missing SCORE line degrades to the verdict alone: PASS
    scores 1.0, FAIL scores 0.0, so a judge that ignores the new line still
    produces a usable reading instead of poisoning the retry budget.
    """
    verdict, reason = _parse_verdict(out)
    m = _SCORE_LINE.search(out or "")
    if m is None:
        return verdict, (1.0 if verdict else 0.0), reason
    score = max(0.0, min(1.0, float(m.group(1)) / 100.0))
    return verdict, score, reason


# One mark per claim, the faithfulness twin of the PASSAGE marks. NONE is
# reserved for the whole-reply case ("the answer makes no factual claims"),
# because that is the only shape in which it is unambiguous.
_CLAIM_MARK = re.compile(
    r"CLAIM\s*(\d+)\s*:\s*(SUPPORTED|UNSUPPORTED|NONE)", re.IGNORECASE
)


def _parse_claim_marks(out: str) -> list[str]:
    """Ordered claim marks from a faithfulness reply: ['SUPPORTED', ...].

    The list index IS the claim number — the caller computes the
    supported/total ratio from it, so the marks must be trustworthy:

    * numbering gaps and duplicated numbers are a truncated or garbled reply
      — a partial claim set would silently inflate the ratio, so JudgeError
      (fail-open to "not graded"), never a score;
    * NONE mixed with real marks is ambiguous (which claims are the none
      ones?) and raises for the same reason.

    Thinking traces are stripped first, exactly as _parse_verdict does, so a
    model rehearsing marks inside <thought> cannot clobber the real ones.
    An empty reply raises via the caller's verdict-anchor parse.
    """
    text = _THINK_BLOCK.sub("", out or "")
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    marks: dict[int, str] = {}
    for m in _CLAIM_MARK.finditer(text):
        idx = int(m.group(1))
        token = m.group(2).upper()
        if idx in marks and marks[idx] != token:
            raise JudgeError(
                f"claim {idx} marked twice: {marks[idx]} then {token}"
            )
        marks[idx] = token
    if not marks:
        return []
    tokens = [marks[k] for k in sorted(marks)]
    if tokens == ["NONE"]:
        return tokens
    if "NONE" in tokens:
        raise JudgeError("claim marks mix NONE with SUPPORTED/UNSUPPORTED")
    if sorted(marks) != list(range(1, len(marks) + 1)):
        raise JudgeError(
            f"claim marks are not a contiguous 1..{len(marks)} set: {tokens}"
        )
    return tokens


def _faithfulness_prompt(question: str, context: str, answer: str) -> str:
    """The shared criteria for both faithfulness judges.

    One prompt body so the benchmark's boolean verdict and the live scorecard's
    ratio always MEAN the same thing — the pairing rule that made the boolean
    twins of every scored judge. "Give no benefit of the doubt" is the whole
    point of the 2026-08-29 tightening: judges default to charity, and a
    charity-defaulted grader reads 100% on answers that asserted facts the
    passages never stated.
    """
    return (
        "You are grading a RAG system for FAITHFULNESS. Decide for each "
        "distinct factual claim in the answer whether the context supports "
        "it.\n"
        "A claim is SUPPORTED only if a passage in the context EXPLICITLY "
        "states it (trivial rewording is fine). Give no benefit of the "
        "doubt: if the context is silent on the claim, only implies it, or "
        "the claim would need outside knowledge to believe, it is "
        "UNSUPPORTED. Hedged wording ('probably', 'likely') does not rescue "
        "an unsupported claim.\n"
        "A hallucinated detail is UNSUPPORTED even when the rest of the "
        "answer is grounded. A figure that contradicts the context is "
        "UNSUPPORTED. Judge from the context only, never outside "
        "knowledge.\n"
        "Judge each claim INDEPENDENTLY on its own merits: a long answer "
        "can be part grounded and part not, and the marks must show that — "
        "one unsupported claim must not colour the marks of the claims "
        "around it.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        + _CLAIMS_CONTRACT
    )


def faithfulness(question: str, context: str, answer: str) -> tuple[bool, str]:
    """Is every claim in the answer supported by the retrieved context?

    This is the missing anti-hallucination check. The judge must decide ONLY
    from the provided context — not from prior knowledge.

    Tightened 2026-08-29 to match its scored sibling: the judge marks each
    claim and the verdict is computed HERE (every claim SUPPORTED, or a
    no-claims refusal), not taken from the model's own PASS/FAIL line — a
    judge asked for one aggregate opinion passed nearly everything. The
    verdict line is the fail-open anchor: a reply without one is a broken
    grader (JudgeError), never a FAIL.
    """
    if not answer.strip():
        return False, "empty answer"
    prompt = _faithfulness_prompt(question, context, answer)
    # 4096, not the 512 default: claim enumeration reasons more than a
    # vibe-read, and the configured judge is thinking-capable — the same
    # budget every scored judge runs with.
    out = _judge(prompt, max_tokens=4096)
    text = _THINK_BLOCK.sub("", out or "")
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    verdict_j, reason = _parse_verdict(text)
    marks = _parse_claim_marks(text)
    if not marks:
        raise JudgeError("no CLAIM marks in judge reply")
    ok = marks == ["NONE"] or all(m == "SUPPORTED" for m in marks)
    if not reason:
        reason = (
            "all claims supported" if ok
            else f"{sum(1 for m in marks if m != 'SUPPORTED')} of "
            f"{len(marks)} claims unsupported"
        )
    # The returned verdict is the MEASURED one; the judge's own line was only
    # the presence anchor. A verdict that disagrees with the marks would be
    # two graders in one column.
    return ok, reason or f"judge verdict: {'PASS' if verdict_j else 'FAIL'}"


def answer_relevancy(question: str, answer: str) -> tuple[bool, str]:
    """Does the answer address the SPECIFIC ask of the question?

    Tightened 2026-08-29: accuracy is not the bar, addressment is. An answer
    that is factually true but answers a different aspect of the question, or
    pads with same-topic background, is a FAIL — the published run's flat
    100% relevancy came from judges that accepted anything on-topic.
    """
    if not answer.strip():
        return False, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER RELEVANCY. Decide whether "
        "the answer addresses the SPECIFIC ask of the question — the "
        "entity, figure, comparison, or scope it requests. An answer that "
        "is factually accurate but answers a DIFFERENT aspect of the "
        "question, restates the question, or offers generic background "
        "instead of the requested fact is FAIL. Off-topic or evasive "
        "answers are FAIL. A specific, direct answer is PASS, and so is an "
        "appropriate 'not found in the documents' when the sources lack "
        "the answer.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


def context_relevance(question: str, context: str) -> tuple[bool, str]:
    """Is the retrieved context materially relevant to the question?

    This is a reference-free live proxy for retrieval precision. It judges the
    passages against the question itself, not against a hidden expected answer,
    so it must never be reported as canonical Context Precision or Recall.
    """
    if not context.strip():
        return False, "empty context"
    prompt = (
        "You are checking the CONTEXT RELEVANCE of passages retrieved for a RAG "
        "question, without a reference answer. PASS when the context contains "
        "useful, directly relevant evidence for the question and has no "
        "substantial irrelevant filler. FAIL when the passages are mostly "
        "unrelated, too weak, or do not bear on what was asked. Judge only the "
        "question and supplied context; do not use outside knowledge and do not "
        "judge the generated answer. This is a reference-free proxy, not a gold "
        "Context Precision or Context Recall score.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


def context_sufficiency(question: str, context: str) -> tuple[bool, str]:
    """Does the context contain enough information to answer the question?

    This is a reference-free live proxy for answer-passage coverage. It asks
    whether the supplied passages are sufficient, not whether every gold passage
    was retrieved, because arbitrary chat questions have no gold passage set.
    """
    if not context.strip():
        return False, "empty context"
    prompt = (
        "You are checking CONTEXT SUFFICIENCY for a RAG question, without a "
        "reference answer. PASS when the supplied passages contain enough "
        "specific information to answer the question completely from the "
        "passages alone. FAIL when key information is absent, the context is "
        "too incomplete, or it cannot support an answer. Judge only the "
        "question and supplied context; do not fill gaps with outside knowledge. "
        "This is a reference-free coverage proxy, not canonical Context Recall.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


# --- scored variants for the live scorecard --------------------------------
#
# Same questions, same criteria, one extra 0-100 reading. The benchmark keeps
# the boolean functions above so its published pass-rate numbers keep meaning
# exactly what they meant when published. Since the pivot to the four canonical
# RAGAS metrics, the live chat calls only faithfulness_scored and
# answer_relevancy_scored from this block (plus context_precision_scored and
# context_recall_scored above it); context_relevance_scored,
# context_sufficiency_scored and answer_correctness_scored are retired from the
# live path but KEPT — their semantics document what the proxies were and were
# not, and deleting them would orphan nothing but would also erase the reason
# the newer judges exist.


def faithfulness_scored(question: str, context: str, answer: str) -> tuple[bool, float, str]:
    """Faithfulness as a ratio of supported claims (RAGAS-style), not a verdict.

    The pass/fail twin is the anti-hallucination gate; this one measures HOW
    MUCH of the answer stands on the context — 3 of 4 claims supported is 75,
    which is a reading a single verdict destroys.

    Tightened 2026-08-29, on the context_precision pattern: the judge no
    longer reports the SCORE itself. It marks each claim SUPPORTED or
    UNSUPPORTED (one line per claim, shared prompt with the boolean judge via
    _faithfulness_prompt) and the ratio is computed HERE in Python — judges
    are unreliable at both arithmetic and vibe-pass/fail, and reliable at
    per-item marks. The verdict line is the fail-open anchor; a reply without
    one, or with marks that do not cover the claims, is a JudgeError ("not
    graded"), never a guessed score.
    """
    if not answer.strip():
        return False, 0.0, "empty answer"
    prompt = (
        _faithfulness_prompt(question, context, answer)
        + "\nSCORE is computed from your marks: SUPPORTED claims divided by "
        "total claims (about half unsupported is 50)."
    )
    # 4096, not 512: the configured judge is thinking-capable and enumerates
    # every claim inside <thought> on long contexts. Probed live: 2048 still
    # ends finish_reason="length" with no verdict emitted; 4096 returns stop
    # with a parseable verdict.
    out = _judge(prompt, max_tokens=4096)
    text = _THINK_BLOCK.sub("", out or "")
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    verdict_j, reason = _parse_verdict(text)
    marks = _parse_claim_marks(text)
    if marks == ["NONE"]:
        # A no-claims answer (a proper refusal) cannot be unfaithful to
        # anything — the RAGAS convention. Answer relevancy judges evasion.
        return True, 1.0, reason or "no factual claims: a proper refusal"
    if not marks:
        raise JudgeError("no CLAIM marks in judge reply")
    supported = sum(1 for m in marks if m == "SUPPORTED")
    score = supported / len(marks)
    # Verdict is the measured one (all claims supported), matching the
    # boolean judge's meaning; the judge's VERDICT line was only the
    # fail-open anchor.
    return score == 1.0, score, reason or f"{supported} of {len(marks)} claims supported"


def answer_relevancy_scored(question: str, answer: str) -> tuple[bool, float, str]:
    """How directly the answer addresses what was asked, 0-100.

    Tightened 2026-08-29 with calibration anchors and a specificity demand:
    the bar is the question's actual ask (entity, figure, comparison, scope),
    not topic overlap. An accurate answer to a different aspect fails, and a
    restatement of the question scores near 0 — previously a true-but-tangent
    answer passed with the same 100 as a direct one.
    """
    if not answer.strip():
        return False, 0.0, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER RELEVANCY. Judge whether "
        "the answer addresses the SPECIFIC ask of the question — the "
        "entity, figure, comparison, or scope it requests.\n"
        "SCORE 100 only for an answer that directly and specifically "
        "answers what was asked. An answer that is accurate but answers a "
        "DIFFERENT aspect of the question scores 40 or below. A "
        "restatement of the question, or generic background instead of the "
        "requested fact, scores near 0. An appropriate 'not found in the "
        "documents' refusal when the sources lack the answer scores 80-95. "
        "VERDICT is PASS unless the answer misses the question.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    # 4096, not 512: the configured judge is thinking-capable and enumerates
    # every passage inside <thought> on long contexts. Probed live: 2048 still
    # ends finish_reason="length" with no verdict emitted; 4096 returns stop
    # with a parseable verdict.
    return _parse_verdict_scored(_judge(prompt, max_tokens=4096))


def context_relevance_scored(question: str, context: str) -> tuple[bool, float, str]:
    """How much of the retrieved context bears on the question, 0-100."""
    if not context.strip():
        return False, 0.0, "empty context"
    prompt = (
        "You are checking the CONTEXT RELEVANCE of passages retrieved for a "
        "RAG question, without a reference answer. SCORE is the percentage of "
        "the context that is useful, directly relevant evidence for the "
        "question; mostly unrelated passages score near 0. Judge only the "
        "question and supplied context; do not use outside knowledge and do "
        "not judge the generated answer. This is a reference-free proxy, not a "
        "gold Context Precision score.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    # 4096, not 512: the configured judge is thinking-capable and enumerates
    # every passage inside <thought> on long contexts. Probed live: 2048 still
    # ends finish_reason="length" with no verdict emitted; 4096 returns stop
    # with a parseable verdict.
    return _parse_verdict_scored(_judge(prompt, max_tokens=4096))


def context_sufficiency_scored(question: str, context: str) -> tuple[bool, float, str]:
    """Whether the passages suffice to answer completely, as a 0-100 coverage."""
    if not context.strip():
        return False, 0.0, "empty context"
    prompt = (
        "You are checking CONTEXT SUFFICIENCY for a RAG question, without a "
        "reference answer. SCORE is how completely the passages cover what is "
        "needed to answer the question: 100 when they suffice entirely, lower "
        "when key information is missing. Judge only the question and supplied "
        "context; do not fill gaps with outside knowledge. This is a "
        "reference-free coverage proxy, not canonical Context Recall.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    # 4096, not 512: the configured judge is thinking-capable and enumerates
    # every passage inside <thought> on long contexts. Probed live: 2048 still
    # ends finish_reason="length" with no verdict emitted; 4096 returns stop
    # with a parseable verdict.
    return _parse_verdict_scored(_judge(prompt, max_tokens=4096))


def context_precision_scored(
    question: str, passages: list[str]
) -> tuple[bool, float, str]:
    """Rank-aware Context Precision (RAGAS-style): average precision over the pool.

    ONE judge call, not one per passage. Each retrieved passage is marked
    RELEVANT/IRRELEVANT in the order it was ranked, and the 0-1 score is
    average precision computed HERE, not by the model — judges are unreliable
    at arithmetic and the score must be reproducible from the marks. Relevant
    material ranking low is what this metric punishes; a flat list of passage
    texts would have thrown that away.

    The judge's VERDICT line is parsed only as the fail-open anchor (no
    verdict → JudgeError → the caller records "not graded"); the verdict we
    RETURN is the measured one: PASS when average precision is at least 0.5.
    Marks that do not cover every passage are a broken or truncated reply —
    a partial mark set would silently distort a rank-aware statistic, so it
    is a JudgeError, never a score.

    ``passages`` must be the retrieved chunks IN POOL ORDER. The live caller
    persists that order at answer time, because the deferred grade request
    has no pool of its own and re-running retrieval would risk grading a
    different list than the model read.
    """
    if not passages:
        return False, 0.0, "no passages retrieved"
    n = len(passages)
    numbered = "\n\n".join(
        f"[PASSAGE {i}]\n{p}" for i, p in enumerate(passages, start=1)
    )
    prompt = (
        "You are grading a RAG system's CONTEXT PRECISION. Below are the "
        "passages its retriever returned, IN THE ORDER IT RANKED THEM. For "
        "EACH passage decide whether it contains information NEEDED to "
        "answer this question — a fact, figure, definition, or statement "
        "that a complete answer would have to use. Mark RELEVANT only for "
        "passages that CLEAR THE BAR: a passage that is same-topic "
        "background, tangential, or merely mentions the subject without "
        "bearing on the specific question is IRRELEVANT. Relevant passages "
        "ranked near the top must yield a high score; relevant passages "
        "buried low are penalized. Judge only the question and the "
        "passages; do not use outside knowledge and do not judge any "
        "generated answer.\n\n"
        f"Question: {question}\n\n"
        f"Passages, in retrieval order:\n\n{numbered}\n\n"
        + _MARKS_CONTRACT
    )
    # 4096, not 512: the configured judge is thinking-capable and reasons about
    # every passage inside <thought> on long pools — the same budget every
    # scored judge runs with.
    out = _judge(prompt, max_tokens=4096)
    text = _THINK_BLOCK.sub("", out or "")
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    verdict_j, reason = _parse_verdict(text)
    marks = {
        int(m.group(1)): m.group(2).upper()
        for m in re.finditer(
            r"PASSAGE\s*(\d+)\s*:\s*(RELEVANT|IRRELEVANT)", text, re.IGNORECASE
        )
    }
    missing = [i for i in range(1, n + 1) if i not in marks]
    if missing:
        raise JudgeError(
            f"judge marked {len(marks)} of {n} passages (missing: {missing[:5]})"
        )
    hits = 0
    precision_sum = 0.0
    for i in range(1, n + 1):
        if marks[i] == "RELEVANT":
            hits += 1
            precision_sum += hits / i
    ap = precision_sum / hits if hits else 0.0
    # The returned verdict is the MEASURED one, not the judge's line: a score
    # and a verdict that disagree would be two graders in one column.
    return ap >= 0.5, ap, reason or f"judge verdict: {'PASS' if verdict_j else 'FAIL'}"


def context_recall_scored(
    question: str, reference: str, context: str
) -> tuple[bool, float, str]:
    """RAGAS Context Recall: how much of a known reference the context supports.

    The reference is broken into its distinct factual claims (silently — the
    judge must not show its working), and SCORE is the percentage of those
    claims the retrieved context states or entails. This is the first live
    reading in this app that measures retrieval against real answer material:
    everything before it judged the context against the question alone,
    which cannot see a passage the ranker dropped.

    An empty reference is a caller bug, not a measurement — the same rule the
    judges apply to an empty judge reply — so it raises JudgeError (fail-open
    to "not graded") instead of manufacturing a 0. An empty CONTEXT is a
    measurement: nothing was retrieved, so nothing of the reference can be in
    it.
    """
    if not reference.strip():
        raise JudgeError("empty reference: there is nothing to attribute to the context")
    if not context.strip():
        return False, 0.0, "empty context"
    prompt = (
        "You are grading a RAG system's CONTEXT RECALL. You are given the "
        "question, a REFERENCE answer, and the CONTEXT the system retrieved. "
        "Silently break the reference into its distinct factual claims, then "
        "decide for each whether the CONTEXT states or entails that same "
        "fact. SCORE is the percentage of the reference's claims the context "
        "supports. VERDICT is PASS when every claim is supported. Judge only "
        "from the context; do not use outside knowledge to fill gaps, and do "
        "not judge the system's generated answer.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer:\n{reference}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    # 4096, not 512: the configured judge is thinking-capable and enumerates
    # every claim inside <thought> on long references — the same budget every
    # scored judge runs with.
    return _parse_verdict_scored(_judge(prompt, max_tokens=4096))


def answer_correctness_scored(
    question: str, expected: str, answer: str
) -> tuple[bool, float, str]:
    """How much of the expected answer's key facts the answer conveys, 0-100."""
    if not answer.strip():
        return False, 0.0, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER CORRECTNESS against an "
        "expected answer. SCORE is the percentage of the expected answer's key "
        "facts the system's answer conveys correctly; wrong facts score lower "
        "than missing ones. Minor phrasing differences are fine. VERDICT is "
        "PASS when the key facts are conveyed.\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"System answer: {answer}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    # 4096, not 512: the configured judge is thinking-capable and enumerates
    # every passage inside <thought> on long contexts. Probed live: 2048 still
    # ends finish_reason="length" with no verdict emitted; 4096 returns stop
    # with a parseable verdict.
    return _parse_verdict_scored(_judge(prompt, max_tokens=4096))


def synthesize_expected(question: str, context: str) -> tuple[str, str]:
    """Draft the expected answer a gold author would have written, from context.

    Live grading of arbitrary questions has no human reference. The honest
    substitute is to DERIVE one from the retrieved passages and label the result
    estimated everywhere it is shown. Two deliberate properties:

    * It reads only the question and the passages. It never sees the system's
      answer, so scoring against it is not the model agreeing with itself.
    * If the passages genuinely cannot answer the question, that is returned as
      a verdict ("no answer derivable"), NOT invented text — grading a real
      answer against a hallucinated reference would manufacture failures out
      of thin air.

    Raises JudgeError when the model cannot produce usable text.
    """
    if not context.strip():
        raise JudgeError("empty context")
    prompt = (
        "You are writing the reference answer for a RAG evaluation set.\n"
        "Answer the question using ONLY the passages below, in 1-3 sentences.\n"
        "If and only if the passages do not contain enough information to "
        "answer, reply with exactly:\n"
        f"{NO_ANSWER_DERIVABLE_SENTINEL.upper()}\n"
        "Do not use outside knowledge. Do not mention the passages or this task.\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{context}\n\n"
        "Reference answer:"
    )
    # Reuses the same client/model as every other judge; raw text here rather
    # than a verdict, so _judge + strip directly. 4096 for the same
    # thinking-budget reason as the scored judges: a draft cut off mid-thought
    # raises "empty content" and reads as an outage.
    out = _judge(prompt, max_tokens=4096)
    text = _THINK_BLOCK.sub("", out or "")
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    if not text:
        raise JudgeError("reference synthesis returned empty content")
    # The refusal token must be the WHOLE reply. A trace that merely quotes it
    # while reasoning toward a real answer is not a refusal; matching anywhere
    # turned a correctly drafted reference into a false "no answer" (found on
    # the preview deployment, 2026-08-27).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].upper() == NO_ANSWER_DERIVABLE_SENTINEL.upper():
        return "", NO_ANSWER_DERIVABLE_SENTINEL
    return text, ""


def answer_correctness(
    question: str, expected: str, answer: str, golden_passages: list[str] | None = None
) -> tuple[bool, str]:
    """Factual correctness vs the expected answer (upgraded from the old string match).

    When golden_passages are supplied (the exact source spans that answer the
    question), the judge may also credit an answer that paraphrases those spans
    correctly even if it differs from the expected wording. This reduces
    mis-flags from phrasing differences.
    """
    if not answer.strip():
        return False, "empty answer"
    golden_block = ""
    if golden_passages:
        golden_block = (
            "\n\nSource passages that answer the question (an answer matching "
            "these in meaning is correct):\n"
            + "\n".join(f"- {p}" for p in golden_passages)
        )
    prompt = (
        "You are grading a RAG system for ANSWER CORRECTNESS. Decide whether the "
        "system's answer conveys the key facts of the expected answer. Minor "
        "phrasing differences are fine; wrong or missing key facts are FAIL."
        f"{golden_block}\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"System answer: {answer}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))
