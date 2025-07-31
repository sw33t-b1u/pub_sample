import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_pw = os.getenv("NEO4J_PASSWORD")
graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pw)
query = input("質問を入力してください: ")


index = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_pw,
    node_label="Device"
    text_node_properties=["name","role","affiliation","device","type","abilities","summary"],
    embedding_node_property="embedding",
    index_name="vector_index", # ベクトル検索用のインデックス名
    keyword_index_name="all_index", # 全文検索用のインデックス名
    search_type="hybrid"
    )

docs_with_score = index.similarity_search_with_score(query, k=2)
for doc in docs_with_score:
    print("回答: ",doc.page_content)

