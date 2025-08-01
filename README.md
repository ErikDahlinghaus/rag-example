# RAG Example

This is a toy example of a simple RAG system.

### Usage:

1. Copy `.env.example` to `.env`
1. Place your markdown documents in `/markdown`
1. From the root of the repo, run `uv run rag-example reprocess`
1. From the root of the repo, run `uv run rag-example query "A question about my data"` to search for relevant documents and display scores
1. From the root of the repo, run `uv run rag-example answer "A question about my data"` to answer that question with the returned documents