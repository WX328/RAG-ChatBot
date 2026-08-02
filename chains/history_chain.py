from operator import itemgetter

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)

from langchain_core.chat_history import (
    InMemoryChatMessageHistory
)

from langchain_core.output_parsers import StrOutputParser

from utils.llm import chat_model



# 保存不同用户聊天记录

store = {}



def get_session_history(session_id):

    if session_id not in store:

        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]



# Prompt

prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",
            """
你是企业知识库助手。

请根据下面的上下文回答问题。

要求：
1. 只根据资料回答
2. 不要编造
3. 如果资料没有答案，请说不知道


上下文:
{context}

"""
        ),


        # 多轮历史

        MessagesPlaceholder(
            variable_name="history"
        ),


        (
            "human",
            "{question}"
        )

    ]

)



def format_docs(docs):

    return "\n\n".join(

        [
            doc.page_content
            for doc in docs
        ]

    )



def create_history_rag_chain(retriever):


    # 普通 RAG Chain

    rag_chain = (

        {

            "context":
            itemgetter("question")
            |
            retriever
            |
            format_docs,


            "question":
            itemgetter("question"),


            # 关键：
            # 给 history 留位置

            "history":
            itemgetter("history")

        }

        |

        prompt

        |

        chat_model

        |

        StrOutputParser()

    )



    # 加入聊天记忆

    history_chain = RunnableWithMessageHistory(

        rag_chain,

        get_session_history,

        input_messages_key="question",

        history_messages_key="history"

    )


    return history_chain