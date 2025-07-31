import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import GraphCypherQAChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_pw = os.getenv("NEO4J_PASSWORD")
graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pw)
embeddings = OpenAIEmbeddings()

hybrid_index = Neo4jVector(
    embedding=embeddings,
    url=neo4j_uri,
    username=neo4j_user,
    password=neo4j_pw,
    search_type="hybrid",
    node_label="Magic",
    text_node_property=["name","abilities","summary"],
    embedding_node_property="embedding",
)

llm = ChatOpenAI(temperature=0, model_name='gpt-4o')
retriever = hybrid_index.as_retriever()
cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=ChatOpenAI(temperature=0),
    graph=graph,
    verbose=False,
    allow_dangerous_requests=True,
)

tools = [
    Tool(
        name="Graph",
        func=cypher_chain.run,
        description="質問の回答が不明だった場合に使用してください。",
    ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False
)

# 質問応答の実行
def ask_question(question):
    response = agent.run(question)
    print(f"Question: {question}")
    print(f"Answer: {response}\n")

# 使用例
question = input("質問を入力: ")
ask_question(question)
