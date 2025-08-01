from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from rag_example.settings import settings


@dataclass
class ChunkedDocument:
    filepath: str
    filename: str
    chunks: list[str]


@lru_cache(maxsize=1)
def get_chunker() -> HybridChunker:
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(settings.EMBEDDING_MODEL),  # type: ignore[no-untyped-call]
        max_tokens=settings.MAX_CHUNK_TOKENS,
    )

    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )


def get_contextualized_chunks(file_path: str) -> ChunkedDocument:
    doc = DocumentConverter().convert(source=file_path).document

    chunker = get_chunker()
    chunk_iter = chunker.chunk(dl_doc=doc)

    contextualized_chunks = []
    for chunk in chunk_iter:
        enriched_text = chunker.contextualize(chunk=chunk)
        contextualized_chunks.append(enriched_text)

    path = Path(file_path)
    return ChunkedDocument(filepath=file_path, filename=path.name, chunks=contextualized_chunks)
