from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()


chat_model = ChatOpenAI(

    model="deepseek-chat",

    temperature=0.2,

    api_key=os.getenv(
        "api_key"
    ),

    base_url="https://api.deepseek.com"

)