from langchain_community.vectorstores import FAISS

from src.embedding import get_embedding
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"



def create_vector_db(chunks):


    embedding=get_embedding()


    db=FAISS.from_documents(
        chunks,
        embedding
    )


    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTOR_DB_DIR))


    return db
