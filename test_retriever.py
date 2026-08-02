from utils.retriever import get_retriever


retriever = get_retriever()


docs = retriever.invoke(
    "请输入一个你的知识库里面肯定存在的问题"
)


print("检索数量:", len(docs))


for i, doc in enumerate(docs):

    print("================")
    print("第", i+1, "条")

    print(
        doc.page_content[:500]
    )