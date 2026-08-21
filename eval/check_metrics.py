"""Deterministic verification of eval/metrics.py (no live proxy needed).

NOT a pytest module, despite what it was called until now: it has no
`test_*` functions, so `pytest` collected nothing from it and reported
"no tests ran" — which reads as a pass. Run it directly:

    .venv/Scripts/python eval/check_metrics.py


Proves the retrieval metrics REACT to retrieval quality, so a good config and
a bad config produce different numbers (spec §9 verification plan). Uses a
trivial bag-of-words embedder so the run is fully deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub ragchat.embeddings so judges.py (imported transitively) doesn't need a key.
import types

fake_embeddings = types.ModuleType("ragchat.embeddings")
sys.modules["ragchat.embeddings"] = fake_embeddings


def _bow(text: str) -> list[float]:
    # Simulate a REAL embedding model: distinct phrases map to distinct,
    # dense, near-orthogonal vectors. Hash the WHOLE phrase to a stable seed
    # and draw a fixed-length random vector. This means a missing/irrelevant
    # phrase has ~0 cosine to the golden passage (unlike a bag-of-words stub
    # where shared tokens leak similarity). For the "good/bad" ranking test we
    # make the relevant phrase IDENTICAL in good/bad (just ranked differently),
    # and "missing" uses a completely different phrase.
    import hashlib, random
    h = hashlib.sha256(text.lower().encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(32)]


class _StubEmb:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return [_bow(t) for t in texts]

    def embed_query(self, text):
        return _bow(text)


fake_embeddings.ProxyEmbeddings = _StubEmb
fake_embeddings.openai_client = lambda: None

import eval.metrics as M

golden = [_bow("kfd client 3a system classify")]
good = [_bow("kfd client 3a system classify"), _bow("project price psf"), _bow("video")]
bad = [_bow("video"), _bow("project price psf"), _bow("kfd client 3a system classify")]
missing = [_bow("video"), _bow("project price psf"), _bow("system")]  # relevant absent

print("=== Context Recall (rank-INSENSITIVE by definition) ===")
print("good   :", round(M.context_recall(good, golden), 3))
print("bad    :", round(M.context_recall(bad, golden), 3), "  (passage present in both => 1.0)")
print("missing:", round(M.context_recall(missing, golden), 3), "  (should be 0.0)")

print("=== Precision@k (k=3) ===")
print("good:", round(M.precision_at_k(good, golden, 3), 3))
print("bad :", round(M.precision_at_k(bad, golden, 3), 3))

print("=== MRR@k (k=3) ===")
print("good:", round(M.mrr_at_k(good, golden, 3), 3), " (rank1)")
print("bad :", round(M.mrr_at_k(bad, golden, 3), 3), " (rank3 => 1/3)")

print("=== NDCG@k (k=3) ===")
print("good:", round(M.ndcg_at_k(good, golden, 3), 3))
print("bad :", round(M.ndcg_at_k(bad, golden, 3), 3))

print("=== Hit Rate@k (k=3) ===")
print("good:", M.hit_rate_at_k(good, golden, 3))
print("bad :", M.hit_rate_at_k(bad, golden, 3))
print("missing:", M.hit_rate_at_k(missing, golden, 3), " (should be 0)")

assert M.mrr_at_k(good, golden, 3) > M.mrr_at_k(bad, golden, 3)
assert M.ndcg_at_k(good, golden, 3) > M.ndcg_at_k(bad, golden, 3)
assert M.context_recall(missing, golden) == 0.0
assert M.hit_rate_at_k(missing, golden, 3) == 0
assert M.hit_rate_at_k(good, golden, 3) == 1
print("\nALL ASSERTIONS PASSED — metrics react correctly to retrieval quality and ranking.")
