from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from chromadb import (
    Collection,
    Documents,
    EmbeddingFunction,
    Embeddings,
    PersistentClient,
    QueryResult,
)
from chromadb.api import ClientAPI
from chromadb.config import Settings
import numpy as np

from rag_example.chunk import get_contextualized_chunks
from rag_example.vector import embed, rerank


@dataclass
class SearchResult:
    document: str
    metadata: dict[str, str | int | float | bool | None]
    id: str
    distance: float
    rerank_score: float


@lru_cache(maxsize=1)
def get_chromadb_client() -> ClientAPI:
    client = PersistentClient(path="./chroma", settings=Settings(allow_reset=True))
    client.heartbeat()
    return client


class ChromaDBEmbeddingFunction(EmbeddingFunction[Documents]):
    def __call__(self, input: Documents) -> Embeddings:
        vectors = embed(input)
        return [np.array(vector, dtype=np.float32) for vector in vectors]


@lru_cache
def get_collection(name: str = "documents") -> Collection:
    collection = get_chromadb_client().get_or_create_collection(
        name=name,
        embedding_function=ChromaDBEmbeddingFunction(),  # type: ignore[arg-type]
    )
    return collection


def write_file_to_chromadb(file_path: str, collection_name: str = "documents") -> None:
    collection = get_collection(collection_name)

    chunked_doc = get_contextualized_chunks(file_path)

    safe_filename = "".join(c if c.isalnum() else "_" for c in chunked_doc.filename)
    ids = [f"{safe_filename}_{i}" for i in range(len(chunked_doc.chunks))]

    metadatas: list[Mapping[str, str | int | float | bool | None]] = [
        {"filename": chunked_doc.filename, "filepath": chunked_doc.filepath, "chunk_index": i}
        for i in range(len(chunked_doc.chunks))
    ]

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=chunked_doc.chunks,
    )


def query_chromadb_text_similarity(query_text: str, n_results: int = 10, collection_name: str = "documents") -> QueryResult:
    """Query ChromaDB with text and return similar chunks."""
    collection = get_collection(collection_name)

    results = collection.query(query_texts=[query_text], n_results=n_results)

    return results


def query_and_rerank(query_text: str, n_results: int = 5, collection_name: str = "documents") -> list[SearchResult]:
    """Query ChromaDB and rerank the results."""
    # Get more results initially for reranking
    initial_results = query_chromadb_text_similarity(query_text, n_results * 2, collection_name)

    if not initial_results["documents"] or not initial_results["documents"][0]:
        return []

    ids = initial_results["ids"][0]
    documents = initial_results["documents"][0]
    metadatas = initial_results["metadatas"][0] if initial_results["metadatas"] else []
    distances = initial_results["distances"][0] if initial_results["distances"] else []

    # Rerank the documents
    reranked_results = rerank(query_text, documents)

    # Create a mapping from document to rerank score
    doc_to_score = {result.document: result.score for result in reranked_results}

    # Combine results with reranked scores
    combined_results = []
    for doc, metadata, doc_id, distance in zip(documents, metadatas, ids, distances, strict=False):
        rerank_score = doc_to_score.get(doc, 0.0)
        combined_results.append(
            SearchResult(
                document=doc,
                metadata=dict(metadata) if metadata else {},
                id=doc_id,
                distance=distance,
                rerank_score=rerank_score,
            )
        )

    # Sort by rerank score (descending) and return top n_results
    combined_results.sort(key=lambda x: x.rerank_score, reverse=True)
    return combined_results[:n_results]
