#https://python.langchain.com/docs/integrations/platforms/openai/
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
question = input("質問を入力してください: ")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

messages = [
    (
        "system",
        "あなたは様々な分野の知識を有した、コンシェルジュです。お客さんからの質問に、日本語で回答してください。",
    ),
    ("human", question),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)

#高町なのは達が使用する魔法デバイスについて教えてください。カートリッジシステムとの違いは何ですか？
#レヴァンティンのシュランゲフォルムのカートリッジ消費量は？
#シュランゲバイセンとは何ですか？
