import streamlit as st

from utils.retriever import get_retriever

from chains.history_chain import create_history_rag_chain



# =====================
# 页面配置
# =====================

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🤖"
)


st.title("🤖 企业知识库智能问答系统")

st.write(
    "基于 LangChain + FAISS + DeepSeek 的 RAG系统"
)



# =====================
# 初始化RAG Chain
# =====================

@st.cache_resource
def load_chain():

    retriever = get_retriever()

    chain = create_history_rag_chain(
        retriever
    )

    return chain



rag_chain = load_chain()



# =====================
# 保存聊天记录
# =====================

if "messages" not in st.session_state:

    st.session_state.messages = []



# 显示历史消息

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



# =====================
# 用户输入
# =====================

question = st.chat_input(
    "请输入你的问题..."
)



if question:


    # 用户消息

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)



    # AI回答

    with st.chat_message("assistant"):

        with st.spinner(
            "正在思考..."
        ):

            answer = rag_chain.invoke(
                {
                    "question": question,
                    "history":[]
                    },
                {
                    "configurable": {
                        "session_id": "user001"
                    }
                }   
            )


            st.markdown(
                answer
            )


    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )