from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path):

    loader = PyPDFLoader(path)

    documents = loader.load()

    return documents



if __name__ == "__main__":

    docs = load_pdf(
        "../data/test.pdf"
    )

    print("文档数量:",len(docs))

    print(
        docs[0].page_content[:500]
    )