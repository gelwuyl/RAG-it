"""Where a citation says the passage came from has to be true.

The excerpt pane shows this string beside the passage, so it is a claim made to
the reader every time they check an answer. It read "~0% of document" next to an
excerpt containing the ENTIRE document — nought percent, and it was all of it.

Two things were wrong at once:

  * the number is where the passage STARTS, but "of document" phrases it as a
    quantity, which is the opposite reading;
  * a chunk that covers its whole document has no position worth reporting, and
    reporting one makes the shortest documents look the most broken.

No network: chunking is a pure function of text and config.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragchat.chunking import Chunk, WHOLE_DOCUMENT_COVERAGE, refine_refs  # noqa: E402


def _chunk(text: str, ref: str = "") -> Chunk:
    return Chunk(text=text, ref=ref)


def test_a_chunk_that_is_the_whole_document_says_so():
    doc = "Meridian Coffee opens at 7:30 AM on weekends. " * 8
    out = refine_refs([_chunk(doc)], doc)
    assert out[0].ref == "whole document", out[0].ref


def test_the_position_is_phrased_as_a_position_not_a_quantity():
    """"~40% of document" reads as "40% of the document" — a share of it. The
    number is an offset, and the words have to agree with the number."""
    doc = "A" * 500 + "the passage we are locating here" + "B" * 500
    out = refine_refs([_chunk("the passage we are locating here")], doc)
    assert out[0].ref.endswith("% through"), out[0].ref
    assert "of document" not in out[0].ref


def test_the_percentage_is_where_the_passage_starts():
    doc = "A" * 250 + "findme" + "B" * 744          # starts at 25%
    out = refine_refs([_chunk("findme")], doc)
    assert out[0].ref == "~25% through", out[0].ref


def test_a_passage_at_the_start_of_a_long_document_still_reads_as_a_position():
    """This is the case that produced the bug: 0 is a legitimate offset, and it
    must not be confused with 'none of the document'."""
    doc = "opening line of the file. " + ("padding. " * 400)
    out = refine_refs([_chunk("opening line of the file.")], doc)
    assert out[0].ref == "~0% through"


def test_the_whole_document_rule_uses_coverage_not_position():
    """A chunk starting at 0 is not the whole document; a chunk containing it
    is. Deciding on position would relabel every first chunk."""
    doc = "x" * 1000
    almost_all = _chunk("x" * int(1000 * WHOLE_DOCUMENT_COVERAGE))
    a_slice = _chunk("x" * 100)
    assert refine_refs([almost_all], doc)[0].ref == "whole document"
    assert refine_refs([a_slice], doc)[0].ref != "whole document"


def test_a_structural_reference_is_never_overwritten():
    """Headings beat offsets: "## Menu Prices" locates a passage better than a
    percentage ever will."""
    doc = "some document text here"
    out = refine_refs([_chunk("text here", ref="## Menu Prices")], doc)
    assert out[0].ref == "## Menu Prices"


def test_a_passage_that_is_not_in_the_document_claims_nothing():
    """Better an empty reference than a confident wrong one."""
    doc = "the document says one thing"
    out = refine_refs([_chunk("this text appears nowhere in it")], doc)
    assert out[0].ref == ""


def test_an_empty_document_does_not_divide_by_zero():
    assert refine_refs([_chunk("anything")], "")[0].ref == ""


# --- and the stored copy has to notice -------------------------------------
#
# The demo template's chunks are copied to every guest verbatim, so a label
# written by old code outlives it. The config fingerprint covers the VECTORS
# and deliberately not this — a citation label has no business invalidating an
# embedding — so the template needs a second signal or it serves the old string
# until some unrelated re-index happens to come along.

def test_the_demo_stamp_changes_with_the_metadata_revision(monkeypatch):
    from ragchat import guests

    before = guests._demo_stamp("some document text")
    monkeypatch.setattr(guests, "CHUNK_METADATA_REVISION",
                        guests.CHUNK_METADATA_REVISION + 1)
    assert guests._demo_stamp("some document text") != before, (
        "bumping the revision must invalidate the template, or a chunk-metadata "
        "change never reaches anyone who already has a copy"
    )


def test_the_demo_stamp_changes_with_the_text():
    from ragchat import guests

    assert guests._demo_stamp("one thing") != guests._demo_stamp("another thing")


def test_the_stamp_is_stable_for_the_same_input():
    """It gates a re-seed. Unstable would mean re-copying the corpus on every
    guest login."""
    from ragchat import guests

    assert guests._demo_stamp("same") == guests._demo_stamp("same")
