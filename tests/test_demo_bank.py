"""The demo bank's ground truth is only ground truth if the data is sound.

``eval/demo_golden.jsonl`` backs the scorecard's gold rows: a matched question
is graded against entries a human curated over the two fixed demo documents.
Three failure modes would quietly turn that gold into fiction, and each gets
its own gate here:

1. A corpus edit that breaks a passage string — the entry would claim passages
   the document no longer contains, so retrieval metrics would measure against
   ghosts. Every passage must be a verbatim substring of its named file.
2. A wording that matches the WRONG question — a near-miss routed to another
   entry would grade an answer against someone else's known answer. Variants
   must sit >= MATCH_RATIO from their own canonical and below it from every
   other question in the bank.
3. A silent rewrite of the original eight entries — their bank identity is
   load-bearing (existing fixtures and published readings lean on it), so the
   bank is append-only and the originals must lead it unchanged.
"""
from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from eval import golden  # noqa: E402

CORPUS_DIR = ROOT / "eval" / "corpus"

# The eight entries the bank shipped with, in file order. Append-only means
# these never move and never change; a test asserting identity (not index)
# keeps future edits honest without coupling to _idx integers.
ORIGINAL_DEMO_8 = [
    "What are the opening hours at Meridian Coffee stores?",
    "How much does oat milk cost at Meridian Coffee?",
    "What temperature should the milk fridges read, and what if they are above 5 degrees?",
    "When does Meridian roast its coffee?",
    "How do I force a firmware update on the SunPak 5?",
    "What warranty does the SunPak 5 ship with?",
    "Does Meridian Coffee offer catering for private events?",
    "How much does the SunPak 5 battery cost?",
]

# Curated expansion: 31 answerable facts, 6 deliberate refusals, plus wording
# variants that share their canonical entry's expected answer and passages.
N_NEW = 83
N_REFUSALS = 15  # 2 original + 6 curated + 7 variants
N_FACTS = 76     # 6 original + 31 curated + 39 variants

# variant question -> its canonical question (whose expected answer and
# passages it must resolve to). Written out rather than derived: a typo here
# fails loudly, while a clever derivation could quietly pass the wrong pair.
VARIANTS_OF = {
    "Which stores does Meridian run and when did each open?": "Which stores does Meridian operate and when did each open?",
    "Which branches does Meridian operate and when did each open?": "Which stores does Meridian operate and when did each open?",
    "Which Meridian store has the roaster, and what kind?": "Which Meridian store houses the roaster, and what kind?",
    "When does the Harbor Point close?": "When does Harbor Point close?",
    "What time do the Meridian stores open on weekends?": "What time do Meridian stores open on weekends?",
    "When do Meridian stores open on weekends?": "What time do Meridian stores open on weekends?",
    "What are the standard menu prices at Meridian?": "What are the standard menu prices at Meridian Coffee?",
    "How does Meridian's loyalty card work?": "How does the Meridian loyalty card work?",
    "What does the Meridian opening checklist start?": "What does the Meridian opening checklist start with?",
    "What boiler pressure should the espresso machine read?": "What boiler pressure should the Meridian espresso machine read?",
    "What boiler pressure should the Meridian espresso machine have?": "What boiler pressure should the Meridian espresso machine read?",
    "How is the grinder calibrated at Meridian Coffee?": "How is the grinder calibrated at Meridian?",
    "What happens when a Meridian milk fridge reads above 5°C?": "What happens if a Meridian milk fridge reads above 5°C?",
    "How much cash do the Meridian registers start with?": "How much cash do Meridian registers start with?",
    "How are cash drops handled at Meridian Coffee?": "How are cash drops handled at Meridian?",
    "When do the Meridian deposits go to the bank?": "When do Meridian deposits go to the bank?",
    "What does Meridian's closing shift do?": "What does the Meridian closing shift do?",
    "How are the Meridian light roasts and house blend roasted?": "How are Meridian light roasts and the house blend roasted?",
    "How should Meridian's green beans be stored?": "How should Meridian green beans be stored?",
    "How much energy does a SunPak 5 store and how much does it weigh?": "How much energy does the SunPak 5 store and how much does it weigh?",
    "What is the SunPak 5's power output rating?": "What is the SunPak 5's output rating?",
    "What are the SunPak 5's output ratings?": "What is the SunPak 5's output rating?",
    "What temperature range can the SunPak 5 handle?": "What temperatures can the SunPak 5 handle?",
    "What temperatures can a SunPak 5 handle?": "What temperatures can the SunPak 5 handle?",
    "What is needed to file a SunPak 5 warranty claim?": "What's needed to file a SunPak warranty claim?",
    "What certification must a SunPak installer hold?": "What certification must SunPak installers hold?",
    "What does mounting the SunPak 5 require?": "What does mounting a SunPak 5 require?",
    "How much clearance does a SunPak 5 need?": "How much clearance does the SunPak 5 need?",
    "How much clearance does the SunPak 5 require?": "How much clearance does the SunPak 5 need?",
    "Which way does the cable enter the SunPak 5?": "Which way does cable enter the SunPak 5?",
    "What is the max cable run from the SunPak inverter?": "What's the maximum cable run from the SunPak inverter?",
    "How often do the SunPak firmware updates arrive?": "How often do SunPak firmware updates arrive?",
    "How long does the forced SunPak update take?": "How long does a forced SunPak update take?",
    "Can you downgrade the SunPak firmware?": "Can you downgrade SunPak firmware?",
    "Can I downgrade SunPak firmware?": "Can you downgrade SunPak firmware?",
    "How fast does Helios Tier 1 support reply?": "How fast does Helios Tier 1 support respond?",
    "What triggers an escalation to the Critical Response line?": "What triggers escalation to the Critical Response line?",
    "What causes escalation to the Critical Response line?": "What triggers escalation to the Critical Response line?",
    "Where is Helios Energy's headquarters?": "Where is Helios Energy headquartered?",
    "Does Meridian offer home delivery?": "Does Meridian offer delivery?",
    "Does Meridian do delivery?": "Does Meridian offer delivery?",
    "Who founded Meridian Coffee Co.?": "Who founded Meridian Coffee?",
    "Does Meridian sell gift cards too?": "Does Meridian sell gift cards?",
    "Does the SunPak 5 work with third party solar panels?": "Does the SunPak 5 work with third-party solar panels?",
    "How much does a SunPak installation cost?": "How much does SunPak installation cost?",
    "Is there a Helios mobile app yet?": "Is there a Helios mobile app?",
}


@pytest.fixture(autouse=True)
def fresh_bank():
    """Re-read the jsonl files: the module caches the bank across tests, and a
    stale cache from another test's monkeypatching would validate the wrong
    data. Reset again on the way out so later tests start clean."""
    golden._bank = None
    yield
    golden._bank = None


def _demo():
    return [e for e in golden.load_bank() if e.get("_src") == "demo"]


def _by_question():
    return {e["question"]: e for e in _demo()}


# ---------- the bank is append-only and complete ----------

def test_original_eight_lead_the_demo_bank_unchanged():
    qs = [e["question"] for e in _demo()[:8]]
    assert qs == ORIGINAL_DEMO_8, "the original demo entries must lead the bank, in order, unedited"


def test_demo_bank_carries_the_curated_expansion():
    demo = _demo()
    assert len(demo) == 8 + N_NEW
    refusals = [e for e in demo if e.get("unanswerable")]
    assert len(refusals) == N_REFUSALS
    assert len(demo) - len(refusals) == N_FACTS


def test_every_variant_wording_is_present_in_the_bank():
    qs = set(_by_question())
    missing = [q for q in VARIANTS_OF if q not in qs]
    assert missing == []
    canon_missing = [c for c in VARIANTS_OF.values() if c not in qs]
    assert canon_missing == []


# ---------- passages are verbatim corpus truth ----------

def test_every_demo_passage_is_verbatim_in_its_corpus_file():
    """Validates the original 8 and every appended entry: a corpus edit that
    breaks any passage string must fail here, not quietly corrupt the gold
    metrics that passage feeds."""
    checked = 0
    for e in _demo():
        if e.get("unanswerable"):
            assert e.get("golden_passages") == []
            continue
        text = (CORPUS_DIR / e["golden_doc"]).read_text(encoding="utf-8")
        for p in e["golden_passages"]:
            assert p in text, (
                f"passage is not a verbatim substring of {e['golden_doc']}: "
                f"{p[:70]!r} (question: {e['question']})"
            )
            checked += 1
    assert checked > 40, "sanity: the bank should carry dozens of passages"


def test_fact_entries_carry_reference_truth_and_refusals_carry_none():
    for e in _demo():
        if e.get("unanswerable"):
            # A refusal asserts the document has NO answer: empty everything.
            assert e.get("expected") == ""
            assert e.get("golden_passages") == []
            assert e.get("needs") == []
            assert e.get("type") == "refusal"
        else:
            assert (e.get("expected") or "").strip(), e["question"]
            assert e.get("golden_passages")
            assert e.get("needs") == ["single_passage"]
            assert e.get("type") == "fact"


# ---------- the matcher routes every question to its own truth ----------

def test_every_new_entry_resolves_to_its_own_truth(monkeypatch):
    """Round-trip: asking each appended question must return an entry with that
    question's own expected answer and passages — never another entry's."""
    original = set(ORIGINAL_DEMO_8)
    for e in _demo():
        if e["question"] in original:
            continue
        m = golden.match_question(e["question"])
        assert m is not None, e["question"]
        assert m["expected"] == e["expected"], e["question"]
        assert m["golden_passages"] == e["golden_passages"], e["question"]
        assert m["unanswerable"] == e["unanswerable"], e["question"]


def test_every_variant_matches_only_its_own_canonical():
    """Each variant sits >= MATCH_RATIO from its canonical and below it from
    every other bank question, and resolves to the canonical's ground truth.
    Never weaken MATCH_RATIO to make a wording pass this gate."""
    by_q = _by_question()
    for vq, cq in VARIANTS_OF.items():
        v, c = golden._norm(vq), golden._norm(cq)
        own = difflib.SequenceMatcher(None, v, c).ratio()
        assert own >= golden.MATCH_RATIO, f"{vq!r} no longer matches its canonical ({own:.3f})"
        for e in _demo():
            oq = e["question"]
            if oq in (vq, cq):
                continue
            r = difflib.SequenceMatcher(None, v, golden._norm(oq)).ratio()
            assert r < golden.MATCH_RATIO, (
                f"{vq!r} is confusable with {oq!r} ({r:.3f})"
            )
        m = golden.match_question(vq)
        canon = by_q[cq]
        assert m["expected"] == canon["expected"], vq
        assert m["golden_passages"] == canon["golden_passages"], vq


def test_no_two_bank_questions_are_below_the_confusion_bar():
    """All-pairs safety net over BOTH bank files: any two questions with
    different identity must sit below MATCH_RATIO, or the matcher could route
    a visitor's wording to the wrong known answer. A pair may clear the ratio
    only when both entries carry the SAME ground truth — expected answer,
    passages, answerability, document — so the routing choice between them
    cannot change what an answer is graded against. (That is the
    canonical/variant relationship; sibling variants are held below the ratio
    by the per-variant test above.) Exact normalized duplicates are banned
    outright: the first would silently win the exact-match race and the second
    would be unreachable."""
    bank = golden.load_bank()

    def _truth(e):
        # What a match hands the grader. Two entries agreeing on all of it are
        # interchangeable; anything else is a wrong-answer routing risk.
        return (
            e.get("expected"),
            tuple(e.get("golden_passages") or ()),
            e.get("unanswerable"),
            e.get("golden_doc"),
        )

    qs = [e["question"] for e in bank]
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            ni, nj = golden._norm(qs[i]), golden._norm(qs[j])
            assert ni != nj, f"exact normalized duplicate: {qs[i]!r} / {qs[j]!r}"
            r = difflib.SequenceMatcher(None, ni, nj).ratio()
            if r < golden.MATCH_RATIO:
                continue
            assert _truth(bank[i]) == _truth(bank[j]), (
                f"confusable pair at {r:.3f}: {qs[i]!r} vs {qs[j]!r} "
                f"carry different ground truth"
            )


def test_match_ratio_is_still_the_strict_ninety():
    assert golden.MATCH_RATIO == 0.90
