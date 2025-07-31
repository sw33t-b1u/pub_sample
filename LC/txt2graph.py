import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import DirectoryLoader

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_pw = os.getenv("NEO4J_PASSWORD")
graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pw)

"""
loader = DirectoryLoader("../original_txt/", glob="*.txt")
docs = loader.load()
len(docs)

"""
original_txt = "../original_txt/nanoha_langchain3.txt"

loader = TextLoader(original_txt)
documents = loader.load()

llm = ChatOpenAI(
    model="gpt-4o", #gpt-4o-mini
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Character", "Device", "Organization","Magic", "Location"],
    allowed_relationships=["FRIENDS_WITH", "ENEMIES_WITH", "USES", "BELONGS_TO", "HAS_ABILITY", "LOCATED_IN","APPEARS_IN", "RELATED"],
    node_properties=["name","role","affiliation","device","type","abilities","summary"],
#    prompt=graph_prompt,
)

graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)
