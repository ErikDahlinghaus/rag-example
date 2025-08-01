from dataclasses import dataclass
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel, FlagReranker  # type: ignore[import-untyped]

from rag_example.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> BGEM3FlagModel:
    model = BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=True)
    return model


def embed(documents: list[str]) -> list[list[float]]:
    vectors = get_embedding_model().encode(documents)
    dense_vectors: list[list[float]] = vectors["dense_vecs"]
    return dense_vectors


@lru_cache(maxsize=1)
def get_reranker_model() -> FlagReranker:
    reranker = FlagReranker(settings.RERANKER_MODEL, use_fp16=True)
    return reranker


@dataclass
class RankedDocument:
    document: str
    score: float


class RerankError(Exception):
    pass


def rerank(query: str, documents: list[str]) -> list[RankedDocument]:
    scores = get_reranker_model().compute_score(
        [(query, document) for document in documents]
    )

    if not scores:
        raise RerankError("scores was None")

    ranked_docs = [
        RankedDocument(document, score)
        for document, score in zip(documents, scores, strict=False)
    ]
    return sorted(ranked_docs, key=lambda x: x.score, reverse=True)
