from langchain_community.embeddings import HuggingFaceBgeEmbeddings



def get_embedding():


    embeddings = HuggingFaceBgeEmbeddings(

        model_name=
        "BAAI/bge-small-zh-v1.5"

    )


    return embeddings
