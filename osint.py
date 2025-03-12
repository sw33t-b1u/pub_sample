from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup

load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("GOOGLE_API_KEY")

# LangGraphのStateを定義
class State(TypedDict):
    query: str  # 入力クエリ
    results: List[str]  # 各エージェントの検索結果
    extracted_urls: List[str]  # 抽出されたURL
    scraped_content: List[str]  # スクレイピングしたコンテンツ
    analysis: Optional[str]  # 解析結果
    final_answer: Optional[str]  # 最終的な解答

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

        # 検索ノード
        def search_phase(state: State) -> State:
            query = state["query"]
            search_results = search_tool.run(query)
            state["results"].append(search_results)
            print("検索ノード:", state["results"])
            return state
        
        # URL抽出ノード
        def extract_urls(state: State) -> State:
            prompt = f"""以下の検索結果から、関連する上位3件のURLを抽出してください。\n\n
                検索結果: {state['results']}\n\n
                出力はリスト形式で記述してください（最大3件）。"""
            response = llm.invoke(prompt)
            extracted_urls = response.content.strip().split("\n")[:3]  # **上位3件のみ取得**
            state["extracted_urls"] = extracted_urls
            print("URL抽出ノード:", extracted_urls)
            return state

        # ウェブスクレイピングノード
        def scrape_content(state: State) -> State:
            scraped_data = []
            for url in state["extracted_urls"]:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")
                        text = soup.get_text(separator=" ", strip=True)
                        scraped_data.append(text[:5000])  # 取得するテキストの長さを制限
                except requests.RequestException as e:
                    print(f"スクレイピングエラー: {e}")
            state["scraped_content"] = scraped_data
            print("スクレイピングノード:", scraped_data)
            return state

        # 解析ノード
        def analyze_phase(state: State) -> State:
            prompt = f"""OSINT問題: {state['query']}\n\n
                検索結果: {state['results']}\n\n 
                これを分析し、必要な情報を抽出してください。"""
            analysis = llm.invoke(prompt)
            state["analysis"] = analysis
            print("解析ノード:", state["analysis"].content)
            return state

        # 回答生成ノード
        def generate_answer(state: State) -> State:
            prompt = f"""OSINT問題: {state['query']}\n\n
                解析結果: {state['analysis']}\n\n 
                これをもとに、問題に対する回答を生成してください。"""
            final_answer = llm.invoke(prompt)
            state["final_answer"] = final_answer
            return state

        def should_continue(state: State) -> str:
            prompt = f"""以下の解析結果から、質問に対する回答として十分な情報が含まれているかを「はい」または「いいえ」で答え、
                もし不十分なら、検索に必要な追加情報を明示してください。\n\n
                解析結果: {state['analysis']}\n\n"""
            response = llm.invoke(prompt)
            response_text = response.content.strip().lower()
            if "いいえ" in response_text:
                missing_info = response_text.replace("いいえ", "").strip()
                state["query"] = f"{state['query']} {missing_info}"  # 不足情報をクエリに追加
                return "search_phase"
            return "end"

        # ノードを追加
        graph.add_node("search_phase", search_phase)
        graph.add_node("extract_urls", extract_urls)
        graph.add_node("scrape_content", scrape_content)
        graph.add_node("analyze_phase", analyze_phase)
        graph.add_node("generate_answer", generate_answer)

        # グラフの接続
        graph.add_edge("search_phase", "extract_urls")
        graph.add_edge("extract_urls", "scrape_content")
        graph.add_edge("scrape_content", "analyze_phase")
        graph.add_edge("analyze_phase", "generate_answer")
        graph.add_conditional_edges(
            "generate_answer", 
            should_continue, 
            {
                "search_phase": "search_phase",  # 不足情報があれば検索に戻る
                "end": END,  # 情報が十分なら終了
            }
        )

        # エントリーポイントの設定
        graph.set_entry_point("search_phase")

        return graph

    def run(self, query: str) -> str:
        """OSINT問題を解く"""
        graph = self.setup_graph().compile()
        initial_state: State = {
            "query": query, 
            "results": [], 
            "extracted_urls": [],
            "scraped_content": [],
            "analysis": None, 
            "final_answer": None
        }
        final_state = graph.invoke(initial_state)
        return final_state["final_answer"]

if __name__ == "__main__":
    osint_agent = OSINTAgent()
    query = input("OSINT問題の入力: ")
    answer = osint_agent.run(query)
    print("最終的な解答:", answer.content)
