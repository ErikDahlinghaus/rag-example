from rag_example.vector import embed, rerank, RankedDocument, vector_db_client, collection

def main():
    # embeddings = embed(["hello_world"])
    # print(embeddings)

    # ranked_documents = rerank("Who is George Washington?", ["A cat", "A US president"])
    # print(ranked_documents)

    # TODO: Chunk documents from markdowns

    # TODO: Insert in to chromadb
    print(collection().upsert(
        ids = ["1", "2"],
        documents = ["hello world", "I am a cat"]
    ))

    print(collection().count())

    print(collection().query(
        query_texts = "I'm a dog"
    ))
    pass


if __name__ == "__main__":
    main()
