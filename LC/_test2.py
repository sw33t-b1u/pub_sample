import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_pw = os.getenv("NEO4J_PASSWORD")


embeddings = OpenAIEmbeddings()

hybrid_db = Neo4jVector(
    embedding=embeddings,
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_pw,
    search_type="hybrid",
    node_label="Magic",
    text_node_property=["name","abilities","summary"],
    embedding_node_property="embedding",
)

retriever = hybrid_db.as_retriever()
chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# 質問応答の実行
query = input("質問を入力してください: ")
result = chain.invoke(
    {"question": query},
    return_only_outputs=True
)

print("回答:", result['answer'])
print("ソース:", result['sources'])