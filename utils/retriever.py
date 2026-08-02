from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from utils.embedding import embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"


def _build_vectorstore():
    pdf_paths = sorted(
        path
        for folder_name in ("data", "documents")
        for path in (PROJECT_ROOT / folder_name).rglob("*.pdf")
        if path.is_file()
    )
    if not pdf_paths:
        raise FileNotFoundError(
            f"没有找到 PDF 文件。请把 PDF 放入 {PROJECT_ROOT / 'data'} 或 "
            f"{PROJECT_ROOT / 'documents'}。"
        )

    documents = []
    for pdf_path in pdf_paths:
        documents.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("PDF 未提取到可索引的文本内容。")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_DB_DIR))
    return vectorstore


def get_retriever():
    index_path = VECTOR_DB_DIR / "index.faiss"
    metadata_path = VECTOR_DB_DIR / "index.pkl"
    if index_path.exists() and metadata_path.exists():
        vectorstore = FAISS.load_local(
            str(VECTOR_DB_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        print("未找到向量索引，正在根据 PDF 首次构建，请稍候...")
        vectorstore = _build_vectorstore()

    return vectorstore.as_retriever(search_kwargs={"k": 3})
