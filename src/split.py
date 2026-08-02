from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )


    chunks = splitter.split_documents(
        docs
    )


    return chunks
