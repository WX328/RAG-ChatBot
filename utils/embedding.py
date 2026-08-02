from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def get_embedding():
    """Return the same local embedding model for indexing and search."""
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


embeddings = get_embedding()
