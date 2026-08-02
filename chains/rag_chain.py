from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.llm import chat_model


prompt = ChatPromptTemplate.from_template(
"""
你是一个企业知识库助手。

请根据下面提供的资料回答问题。

如果资料中没有答案，请回答：
"没有找到相关信息"


资料:
{context}


问题:
{question}

"""
)



def format_docs(docs):

    print("\n==========检索结果==========")

    print("文档数量:", len(docs))


    for i, doc in enumerate(docs):

        print("\n第", i+1, "条:")

        print(
            doc.page_content[:300]
        )


    print("============================\n")


    return "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )



def create_rag_chain(retriever):


    rag_chain = (

        {
            "context":
            retriever | format_docs,


            "question":
            RunnablePassthrough()
        }

        |

        prompt

        |

        # DeepSeek模型
        chat_model

        |

        StrOutputParser()

    )


    return rag_chain