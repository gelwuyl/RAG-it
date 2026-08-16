# RAG Chat — Idea

A multi-user web app where you chat with an LLM that answers **only from your
own documents** (PDFs, web pages, Markdown/text) and cites its sources.

## The idea

- **Grounded answers** — every answer comes from your uploaded documents,
  never the model's general knowledge.
- **Citations** — answers link back to the exact source passage.
- **Multi-format ingestion** — upload PDF/text/Markdown, paste a URL, or sync
  a folder of files.
- **Tunable pipeline** — chunking, retrieval, and generation are set in
  `config.yaml` (and a live Settings panel) with no code changes.
- **Measurable quality** — a live LLM-judge line under each answer
  (similarity, faithfulness, relevancy, latency); an offline golden-set eval
  harness is planned next.
- **Multi-user, isolated** — each account's documents and chats are fully
  separated.
- **Built to grow** — clean module boundaries so a larger product can sit on
  top.

## How it works (in words)

The browser UI (vanilla JS + Vite) talks to a FastAPI backend. On upload,
documents are loaded, split into chunks, embedded through the class LLM
proxy, and stored in a Neon Postgres database using pgvector. On a question,
the app rewrites follow-ups into a standalone query, retrieves the top chunks
(vector search plus BM25 keyword fusion, with an optional reranker), and
generates an answer grounded in those chunks. Sign-in is local
username/password with optional Google OAuth. It deploys on Vercel behind a
single port, with the frontend proxying `/api` to the backend.

## Current defaults

- Generation model: `models/gemma-4-26b-a4b-it`
- Embeddings model: `models/gemini-embedding-001` (768-dim)
- Both served through the class LLM proxy (OpenAI-compatible `/v1`)

## Not yet built

Offline golden-set eval harness, streaming answers, and retrieval upgrades
(Contextual Retrieval, HyDE, multi-query, parent-child).
