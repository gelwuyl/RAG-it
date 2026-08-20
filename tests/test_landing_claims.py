"""The landing page advertises numbers. They have to be true.

"56 golden questions over a 27-document corpus" is a claim made to every
visitor before they have seen anything, and it went stale once already — the
page said 10 documents for as long as the corpus had 27, because nothing
connected the sentence to the directory it describes.

Counting them here is the cheap half of the fix. The expensive half would be
generating the page from the data, which is not worth a build step for four
numbers that change once a quarter; a failing test naming the file to edit is.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LANDING = ROOT / "frontend" / "index.html"
CORPUS = ROOT / "eval" / "corpus"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"


def _landing() -> str:
    return LANDING.read_text(encoding="utf-8")


def _n_corpus() -> int:
    return len([p for p in CORPUS.iterdir() if p.is_file()])


def _n_golden() -> int:
    return len([ln for ln in GOLDEN.read_text(encoding="utf-8").splitlines() if ln.strip()])


def test_the_tiles_count_what_is_actually_there():
    tiles = [int(n) for n in re.findall(r'tile-num">(\d+)<', _landing())]
    assert tiles, "the by-the-numbers tiles are gone or renamed"
    assert _n_golden() in tiles, (
        f"landing page tiles {tiles} do not include the real golden-set size "
        f"({_n_golden()}) — update frontend/index.html"
    )
    assert _n_corpus() in tiles, (
        f"landing page tiles {tiles} do not include the real corpus size "
        f"({_n_corpus()}) — update frontend/index.html"
    )


def test_the_prose_claim_matches_too():
    """The tiles and the sentence drifted apart once; both are read."""
    m = re.search(r"(\d+) golden questions over a (\d+)-document corpus", _landing())
    assert m, "the corpus claim was reworded — re-point this test at it"
    assert int(m.group(1)) == _n_golden(), (
        f"claims {m.group(1)} golden questions, there are {_n_golden()}"
    )
    assert int(m.group(2)) == _n_corpus(), (
        f"claims {m.group(2)} corpus documents, there are {_n_corpus()}"
    )


def test_the_vector_dimension_claim_matches_the_invariant():
    """768 is load-bearing, not decoration: the Neon chunks table has ONE fixed
    vector(768) column shared by every model (CLAUDE.md)."""
    from ragchat.embeddings import embedding_dim

    tiles = [int(n) for n in re.findall(r'tile-num">(\d+)<', _landing())]
    assert embedding_dim() in tiles, (
        f"the page advertises {tiles} dimensions, embedding_dim() returns "
        f"{embedding_dim()}"
    )


# --- the README makes the same kind of claim ------------------------------
#
# The stack table names the two models that are actually shipped. That went
# stale within one commit: config.yaml was reverted from flash-lite back to
# gemma and the table kept advertising flash-lite, because nothing connected
# the sentence to the file it describes. Same fix as the tiles above.


def _shipped(key: str) -> str:
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return cfg["generation"]["llm_model"] if key == "llm" else cfg["embedding"]["model"]


def _readme_row(label: str) -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(rf"^\|\s\*\*{label}\*\*\s\|([^|]*)\|", text, re.MULTILINE)
    assert m, f"the README stack table has no **{label}** row any more"
    return m.group(1)


def test_the_readme_names_the_model_that_actually_ships():
    model = _shipped("llm")
    cell = _readme_row("Generation")
    assert model in cell, (
        f"README advertises {cell.strip()!r} but config.yaml ships {model!r} "
        f"— update the Generation row in README.md"
    )


def test_the_readme_names_the_embedding_model_that_actually_ships():
    model = _shipped("emb")
    cell = _readme_row("Embeddings")
    assert model in cell, (
        f"README advertises {cell.strip()!r} but config.yaml ships {model!r} "
        f"— update the Embeddings row in README.md"
    )
