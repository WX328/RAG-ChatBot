from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)


response = client.chat.completions.create(

    model="deepseek-chat",

    messages=[
        {
            "role":"user",
            "content":"介绍一下RAG技术"
        }
    ]

)


print(response.choices[0].message.content)