"""Named pipeline configurations offered in Settings (PRODUCT_UX_PLAN.md §7).

This is the SINGLE source of the values. They used to live in `frontend/app.js`,
which was fine until the eval harness needed to measure them: a preset the
benchmark scores has to be the same preset the UI ships, or the numbers describe a
configuration nobody can select. The frontend fetches these from
``GET /api/presets``; ``eval/run_eval.py --preset <id>`` runs against them.

What a preset does NOT touch: the model and provider fields. Those are deployment
facts rather than tradeoffs — switching embedding provider re-points every vector
in the store and 422s outright when that provider has no key — so a card called
"Fast" has no business deciding them. Nor keyword fusion, which is unconditional
(see config.load_config).

`values` keys must all exist on PipelineConfig, since the eval harness applies
them with dataclasses.replace.
"""
from __future__ import annotations

# Keys a preset sets. Also the keys the UI compares against the live config to
# decide which preset (if any) the current settings match.
PRESET_KEYS = (
    "chunk_size",
    "chunk_overlap",
    "splitter",
    "top_k",
    "candidate_k",
    "similarity_threshold",
    "reranker",
    "query_rewrite",
    "temperature",
)

# Keys whose change invalidates existing chunks — the same set the config update
# route uses to decide `needs_reindex`, minus the embedding fields no preset
# touches. The UI badges a card with "needs re-index" by comparing these against
# what is actually saved.
INDEX_KEYS = ("chunk_size", "chunk_overlap", "splitter")

PRESETS = [
    {
        "id": "fast",
        "name": "Fast",
        "desc": "One model call per question. No rerank, no rewrite, a small pool — the quickest answer this pipeline can give.",
        "values": {
            "chunk_size": 512,
            "chunk_overlap": 75,
            "splitter": "recursive",
            "top_k": 3,
            "candidate_k": 10,
            "similarity_threshold": 0.0,
            "reranker": False,
            "query_rewrite": False,
            "temperature": 0.0,
        },
    },
    {
        "id": "balanced",
        "name": "Balanced",
        # Reranking used to be the thing that separated this from High accuracy,
        # which made "the shipped default" a lie once the default config turned
        # the reranker on. It is affordable here because the reranker is Cohere
        # rerank-v3.5 now — ONE call for the whole pool, not one per chunk.
        "desc": "The shipped default. Follow-ups are rewritten so they still retrieve, and one rerank pass sharpens the order.",
        "values": {
            "chunk_size": 512,
            "chunk_overlap": 75,
            "splitter": "recursive",
            "top_k": 4,
            "candidate_k": 20,
            "similarity_threshold": 0.0,
            "reranker": True,
            "query_rewrite": True,
            "temperature": 0.0,
        },
    },
    {
        "id": "accurate",
        "name": "High accuracy",
        # No longer "the one that reranks" — Balanced does too. What is left is
        # the shape of the net: smaller chunks, twice the pool, more passages kept.
        "desc": "Smaller chunks and a 40-wide pool, with six passages kept instead of four. Best measured recall, and the slowest to answer.",
        "values": {
            "chunk_size": 384,
            "chunk_overlap": 96,
            "splitter": "recursive",
            "top_k": 6,
            "candidate_k": 40,
            "similarity_threshold": 0.0,
            "reranker": True,
            "query_rewrite": True,
            "temperature": 0.0,
        },
    },
    {
        "id": "low-cost",
        "name": "Low cost",
        "desc": "Bigger chunks means roughly a third fewer embedding calls to index, and the smallest prompt per question.",
        "values": {
            "chunk_size": 768,
            "chunk_overlap": 64,
            "splitter": "recursive",
            "top_k": 3,
            "candidate_k": 8,
            "similarity_threshold": 0.0,
            "reranker": False,
            "query_rewrite": False,
            "temperature": 0.0,
        },
    },
]


def preset_ids() -> list[str]:
    return [p["id"] for p in PRESETS]


def get_preset(preset_id: str) -> dict | None:
    for p in PRESETS:
        if p["id"] == preset_id:
            return p
    return None
