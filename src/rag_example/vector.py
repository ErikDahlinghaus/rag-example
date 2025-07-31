from functools import lru_cache
from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from chromadb import PersistentClient, Documents, Embeddings, EmbeddingFunction


@lru_cache(maxsize=1)
def embedding_model() -> BGEM3FlagModel:
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    return model


def embed(documents: str | list[str]) -> list[float] | list[list[float]]:
    dense_vectors = embedding_model().encode(documents)['dense_vecs']
    return dense_vectors


@lru_cache(maxsize=1)
def reranker_model() -> FlagReranker:
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    return reranker


@dataclass
class RankedDocument:
    document: str
    score: float

def rerank(query: str, documents: list[str]) -> list[RankedDocument]:
    scores = reranker_model().compute_score([[query, document] for document in documents])
    ranked_docs = [RankedDocument(document, score) for document, score in zip(documents, scores)]
    return sorted(ranked_docs, key=lambda x: x.score, reverse=True)


@lru_cache(maxsize=1)
def vector_db_client():
    client = PersistentClient(path="./chroma")
    client.heartbeat()
    return client


class ChromaDBEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        vectors = embed(input)
        return vectors
    

@lru_cache
def collection(name: str = "documents"):
    collection = vector_db_client().get_or_create_collection(
        name=name,
        embedding_function=ChromaDBEmbeddingFunction()
    )
    return collection