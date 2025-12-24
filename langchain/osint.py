from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph
from IPython.display import Image, display

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("GOOGLE_API_KEY")

# LangGraphのStateを定義
class State(TypedDict):
    query: str  # 入力クエリ
    search_query: str  # 検索用クエリ
    results: List[str]  # 検索結果
    final_answer: Optional[str]  # 最終的な解答
    attempt_count: int

# Google検索ツール
search = GoogleSearchAPIWrapper()
search_tool = Tool(name="Google Search", func=search.run, description="Googleを使って情報を検索する")

# https://platform.openai.com/docs/pricing
llm = ChatOpenAI(model_name="gpt-4o-mini")

# LangGraphエージェントの定義
class OSINTAgent:
    def __init__(self):
        self.graph = StateGraph(State)

    def setup_graph(self):
        """LangGraphを使ったエージェントのフローを定義"""
        graph = self.graph

        def generate_search_query(state: State) -> State:
            prompt = f"""ユーザーの質問を適切な検索キーワードに変換してください。\n\n
                質問: {state['query']}\n\n
                出力: 検索エンジンに適したキーワードのみを返してください。"""
            response = llm.invoke(prompt)
            state["search_query"] = response.content.strip()
            print("検索クエリ生成:\n", state["search_query"])
            return state

        # 検索ノード
        def search_phase(state: State) -> State:
            query = state["search_query"]
            search_results = search_tool.run(query)
            state["results"].append(search_results)
            print("検索ノード:\n", state["results"])
            return state

        # 回答生成ノード
        def generate_answer(state: State) -> State:
            prompt = f"""問題: {state['query']}\n\n
                検索結果: {state['results']}\n\n 
                これをもとに、問題に対する回答を生成してください。"""
            response = llm.invoke(prompt)
            state["final_answer"] = response.content
            return state

        def should_continue(state: State) -> str:
            if state.get("attempt_count", 0) >= 2:
                prompt = f"""問題に対する回答を、検索結果を参考に更新してください。
                    問題: {state['query']}\n\n
                    回答: {state['final_answer']}\n\n
                    検索結果: {state['results']}
                """
                response = llm.invoke(prompt) 
                state["final_answer"] = response.content
                return "end"
            
            prompt = f"""問題と回答を確認し、内容に過不足がないか確認してください。
                過不足がある場合は、検索に必要な情報を「不足しています。追加で必要な情報は〜〜です」という形式で回答してください。\n
                問題: {state['query']}\n\n
                回答: {state['final_answer']}\n\n
            """
            response = llm.invoke(prompt)
            response_text = response.content.strip().lower()

            if "不足しています。" in response_text:
                missing_info = response_text.replace("不足しています。", "").strip()
                state["query"] = f"{state['query']} {missing_info}"  # 不足情報をクエリに追加
                print("不足情報を追加:\n", missing_info)
                return "generate_search_query"
            return "end"

        # ノードを追加
        graph.add_node("generate_search_query", generate_search_query)
        graph.add_node("search_phase", search_phase)
        graph.add_node("generate_answer", generate_answer)

        # グラフの接続
        graph.add_edge("generate_search_query", "search_phase")
        graph.add_edge("search_phase", "generate_answer")
        graph.add_conditional_edges(
            "generate_answer", 
            should_continue, 
            {
                "generate_search_query": "generate_search_query",  # 不足情報があれば検索に戻る
                "end": END,  # 情報が十分なら終了
            }
        )

        # エントリーポイントの設定
        graph.set_entry_point("generate_answer")
        return graph

    def run(self, query: str) -> str:
        """OSINT問題を解く"""
        graph = self.setup_graph().compile()
        #可視化
        mermaid_png = graph.get_graph().draw_mermaid_png()
        output_path = "graph.png"
        with open(output_path, "wb") as f:
            f.write(mermaid_png)

        initial_state: State = {
            "query": query, 
            "search_query": "",
            "results": [], 
            "final_answer": None
        }
        final_state = graph.invoke(initial_state)
        return final_state["final_answer"]

if __name__ == "__main__":
    osint_agent = OSINTAgent()
    query = input("OSINT問題の入力: ")
    answer = osint_agent.run(query)
    print("最終的な解答:", answer)
