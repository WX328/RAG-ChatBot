from utils.llm import chat_with_llm
from utils.retriever import get_retriever



# 1. 获取向量数据库检索器

retriever = get_retriever()



def rag_answer(question):

    # ======================
    # 1. 检索相关文档
    # ======================

    docs = retriever.get_relevant_documents(
        question
    )


    # ======================
    # 2. 拼接上下文
    # ======================

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    # ======================
    # 3. 构造Prompt
    # ======================

    prompt = f"""

你是一个专业AI助手。

请根据下面资料回答问题。


资料:
{context}


问题:
{question}


要求:
1. 只能根据资料回答
2. 不要编造答案
3. 如果资料没有答案，请回答不知道

"""


    # ======================
    # 4. 调用大模型
    # ======================

    answer = chat_with_llm(prompt)


    return answer



if __name__ == "__main__":


    question =input( "请输入你的问题")


    result = rag_answer(question)


    print("\nAI回答:")
    print(result)