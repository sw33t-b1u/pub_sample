import os, json, uuid, logging, re, shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import fitz
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import hashlib
import google.auth
from google.auth.transport.requests import Request
from google.cloud import spanner
from google.cloud import storage
from google import genai
from google.genai import types as genai_types
import time

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="CTI Multi-Agent System")

# ---------------------------------------------------------
# 1. Env / Configuration
# ---------------------------------------------------------
PROJECT_ID = os.environ.get("PROJECT_ID")
SPANNER_INSTANCE = os.environ.get("SPANNER_INSTANCE_ID") 
SPANNER_DATABASE = os.environ.get("SPANNER_DATABASE_ID") 
SPANNER_PROJECT_ID = os.environ.get("SPANNER_PROJECT_ID")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
REGION = os.environ.get("REGION")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION")
RAG_CORPUS_RESOURCE = os.environ.get("RAG_CORPUS_RESOURCE")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET")

# Gemini Client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=VERTEX_LOCATION)

# Spanner Client
spanner_client = spanner.Client(project=SPANNER_PROJECT_ID)
try:
    spanner_db = spanner_client.instance(SPANNER_INSTANCE).database(SPANNER_DATABASE)
except Exception as e:
    logging.error(f"Spanner init warning (ignore if building): {e}")
    spanner_db = None

TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------

def get_access_token() -> str:
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not creds.valid:
        creds.refresh(Request())
    return creds.token

def rag_search_via_vertex(query: str, top_n: int = 30, *, run_id: str | None = None, lang: str | None = None) -> List[Dict[str, Any]]:
    """Vertex AI Search RAG Engine retrieveContexts を呼び出す。失敗は例外で返す。"""
    token = get_access_token()
    url = f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{VERTEX_LOCATION}:retrieveContexts"
    payload = {
        "query": {
            "text": query,
            "ragRetrievalConfig": {"topK": top_n},
        },
        "vertexRagStore": {
            "ragResources": [{"ragCorpus": RAG_CORPUS_RESOURCE}],
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    t0 = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        logging.exception(f"[RAG] request failed run_id={run_id} lang={lang} err={e}")
        raise RagRetrievalError(f"RAG request failed: {e}")

    latency_ms = int((time.time() - t0) * 1000)

    if resp.status_code != 200:
        body = resp.text[:500]
        logging.error(
            f"[RAG] non-200 run_id={run_id} lang={lang} status={resp.status_code} latency_ms={latency_ms} body_head={body}"
        )
        raise RagRetrievalError(
            f"RAG retrieveContexts failed with status {resp.status_code}",
            status_code=resp.status_code,
            body=body,
        )

    try:
        data = resp.json()
    except Exception as e:
        body = resp.text[:500]
        logging.error(f"[RAG] invalid json run_id={run_id} lang={lang} latency_ms={latency_ms} body_head={body}")
        raise RagRetrievalError(f"RAG response is not valid JSON: {e}", status_code=200, body=body)

    contexts = data.get("contexts")

    if isinstance(contexts, list):
        contexts_list = contexts
    elif isinstance(contexts, dict) and isinstance(contexts.get("contexts"), list):
        contexts_list = contexts["contexts"]
    else:
        logging.warning(f"[RAG] empty/missing contexts run_id={run_id} lang={lang}")
        return []

    chunks: List[Dict[str, Any]] = []
    for i, c in enumerate(contexts_list, start=1):
        if not isinstance(c, dict):
            continue
        chunks.append({
            "rank": i,
            "text": (c.get("text") or ""),
            "source_uri": c.get("sourceUri"),
        })
    return chunks

def save_to_gcs(run_id: str, content: str):
    """生成されたレポートをMarkdownとしてGCSに保存する"""
    if not OUTPUT_BUCKET:
        logging.warning("OUTPUT_BUCKET is not set. Skipping GCS save.")
        return
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(OUTPUT_BUCKET)
        blob = bucket.blob(f"reports/{run_id}.md")
        blob.upload_from_string(content, content_type="text/markdown")
        logging.info(f"Report saved to GCS: gs://{OUTPUT_BUCKET}/reports/{run_id}.md")
    except Exception as e:
        logging.error(f"Failed to save to GCS: {e}")

def _stable_uuid(prefix: str, value: str) -> str:
    """
    値から安定したIDを生成（INSERT OR IGNORE が効くようにする）
    """
    key = f"{prefix}:{value.strip().lower()}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def _extract_json_array(text: str) -> list:
    """
    Geminiが余計な前後文を付けたり、JSONが崩れた場合に備えた救済。
    期待: [...] の配列
    """
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e != -1 and e > s: return json.loads(text[s : e + 1])
    except: return []
    return []

def _extract_json_object(text: str) -> dict:
    """
    Geminiが余計な前後文を付けたり、JSONが崩れた場合に備えた救済。
    期待: {...} のオブジェクト
    """
    if not text: return {}
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except: pass
    try:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            v = json.loads(text[s : e + 1])
            return v if isinstance(v, dict) else {}
    except: return {}
    return {}

def _select_diverse_chunks(chunks: List[Dict[str, Any]], *, target_k: int = 10, max_chars: int = 1500, max_per_source: int = 2) -> List[Dict[str, Any]]:
    """
    - source_uri で多様性を確保
    - まず各 source_uri から max_per_source まで採用
    - それでも足りなければランキング順に埋める
    - text は max_chars に切る
    """
    if not chunks:
        return []
    ranked = sorted(chunks, key=lambda x: x.get("rank", 10**9))
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for c in ranked:
        src = (c.get("source_uri") or "unknown").strip() or "unknown"
        by_src.setdefault(src, []).append(c)

    selected: List[Dict[str, Any]] = []
    # 1st pass: 各ソースから max_per_source まで
    for src, items in by_src.items():
        for c in items[:max_per_source]:
            selected.append(c)
            if len(selected) >= target_k:
                break
        if len(selected) >= target_k:
            break

    # 2nd pass: 足りない分をランキング順に埋める（重複は避ける）
    if len(selected) < target_k:
        selected_ids = set(id(x) for x in selected)
        for c in ranked:
            if id(c) in selected_ids:
                continue
            selected.append(c)
            if len(selected) >= target_k:
                break

    # 整形（rank 振り直し、text 切り詰め）
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(selected[:target_k], start=1):
        out.append({
            "rank": i,
            "text": (c.get("text") or "")[:max_chars],
            "source_uri": c.get("source_uri"),
        })
    return out

def _evidence_gate(plan: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    根拠不足のときは生成を中止し、理由を返す
    """
    needs = plan.get("evidence_requirements") or {}
    must_have_actor = bool(needs.get("must_have_actor", True))
    must_have_ttp = bool(needs.get("must_have_ttp", False))
    min_sources = int(needs.get("min_unique_sources", 2))
    min_chunks = int(needs.get("min_chunks", 3))

    reasons: List[str] = []
    if not chunks or len(chunks) < min_chunks:
        reasons.append(f"証拠チャンク数が不足（取得 {len(chunks)} < 必要 {min_chunks}）")

    uniq = set((c.get("source_uri") or "unknown") for c in (chunks or []))
    if len(uniq) < min_sources:
        reasons.append(f"参照ソースの多様性が不足（ユニーク {len(uniq)} < 必要 {min_sources}）")

    # Actor/TTP の “痕跡” チェック（厳密同定は後続の抽出・更新でやる）
    joined = "\n".join((c.get("text") or "") for c in (chunks or []))
    if must_have_actor:
            # 判定キーワードを追加（Malware family名なども許容したい場合は適宜追加）
            if not re.search(r"\bAPT\b|threat actor|攻撃者|アクター|Group|Campaign", joined, flags=re.IGNORECASE):
                reasons.append("攻撃者（Actor）に関する根拠が見当たらない（APT/Actor等の記述が不足）")

    if must_have_ttp:
        # ID(Txxxx)だけでなく、攻撃手法を表すキーワードも許容する
        has_id = re.search(r"\bT\d{4}(?:\.\d{3})?\b", joined)
        has_keyword = re.search(r"Phishing|Exploit|Vulnerability|Malware|Ransomware|Backdoor|C2|Command and Control|フィッシング|脆弱性|マルウェア", joined, flags=re.IGNORECASE)
        
        if not (has_id or has_keyword):
            reasons.append("TTP（MITRE ID または 攻撃手法の記述）が根拠内に見当たらない")

    ok = len(reasons) == 0
    return {"ok": ok, "reasons": reasons, "unique_sources": len(uniq), "chunks": len(chunks)}

def _insufficient_report(req: Dict[str, Any], plan: Dict[str, Any], gate: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    """
    根拠不足時に返すMarkdown
    """
    lines = []
    lines.append("# 根拠不足のため回答を生成できません")
    lines.append("")
    lines.append("## 依頼内容")
    lines.append(f"- クエリ: {req.get('query')}")
    lines.append(f"- 読者: {req.get('audience')}")
    lines.append(f"- 期間: {req.get('time_range')}")
    lines.append(f"- 業界: {req.get('industry')}")
    lines.append("")
    lines.append("## 判定されたインテリジェンスレベル")
    lines.append(f"- {plan.get('intent_level')}")
    lines.append("")
    lines.append("## 不足している根拠")
    for r in gate.get("reasons") or []:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## 取得できた証拠（参考：抜粋）")
    for c in chunks[:10]:
        src = c.get("source_uri") or "unknown"
        text_head = (c.get("text") or "")[:120]
        text_head = text_head.replace("\n", " ")
        lines.append(f"- [{c.get('rank')}] source={src} / text_head={text_head}")
    lines.append("")
    return "\n".join(lines)

def load_from_gcs(bucket_name: str, file_path: str) -> str:
    """GCSからMarkdownレポートの内容を読み込む"""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        content = blob.download_as_text()
        return content
    except Exception as e:
        logging.error(f"Failed to load from GCS: {e}")
        raise HTTPException(status_code=404, detail=f"File not found in GCS: {e}")

# ---------------------------------------------------------
# 3. Models
# ---------------------------------------------------------
class AgentRunRequest(BaseModel):
    query: str
    audience: Optional[str] = "Engineer"
    time_range: Optional[str] = "last 30 days"
    industry: Optional[str] = "All"

class AgentRunResponse(BaseModel):
    run_id: str
    intent_level: str
    markdown: str
    review_status: bool
    feedback: Optional[str]


class RagRetrievalError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

class UpdateGraphRequest(BaseModel):
    bucket_name: str
    file_path: str  # 例: "reports/536be36b-cf9a-4881-9eea-9f2cb5b4e5b0.md"

# ---------------------------------------------------------
# 4. Agents Implementation
# ---------------------------------------------------------

class QueryPlannerAgent:
    """
    ユーザ入力を元に:
    - intent_level（STRATEGIC / TACTICAL / OPERATIONAL）
    - search_query（RAGへ投げる検索用クエリ）
    - evidence_requirements（根拠ゲート要件）
    をJSONで返す
    """
    def plan(self, req: AgentRunRequest) -> Dict[str, Any]:
        prompt = f"""
        あなたは脅威インテリジェンスのプランナーです。
        次のユーザ入力から、検索計画をJSONで出力してください（JSONオブジェクトのみ）。

        ユーザ入力:
        - query: {req.query}
        - audience: {req.audience}
        - time_range: {req.time_range}
        - industry: {req.industry}

        出力JSONスキーマ:
        {{
        "intent_level": "STRATEGIC|TACTICAL|OPERATIONAL",
        "search_queries": {{
            "ja": "RAG検索に使う日本語クエリ（ユーザ入力を十分反映）",
            "en": "RAG検索に使う英語クエリ（ユーザ入力を十分反映）"
        }},
        "evidence_requirements": {{
            "min_unique_sources": 2,
            "min_chunks": 3,
            "must_have_actor": true,
            "must_have_ttp": true|false
        }}
        }}
        """.strip()

        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            plan = _extract_json_object(resp.text)
        except Exception as e:
            logging.error(f"QueryPlannerAgent.plan failed: {e}")
            plan = {}

        # フォールバック
        intent = (plan.get("intent_level") or "TACTICAL").strip().upper()
        if intent not in ("STRATEGIC", "TACTICAL", "OPERATIONAL"):
            intent = "TACTICAL"
        plan["intent_level"] = intent

        # search_queries（ja/en）が無い場合のフォールバック
        sq_obj = plan.get("search_queries")
        if not isinstance(sq_obj, dict): sq_obj = {}
        sq_ja = (sq_obj.get("ja") or "").strip()
        sq_en = (sq_obj.get("en") or "").strip()
        if not sq_ja: sq_ja = f"{req.query} 読者:{req.audience} 期間:{req.time_range}"
        if not sq_en: sq_en = f"{req.query} Japan targeted threat actors TTP MITRE ATT&CK"
        plan["search_queries"] = {"ja": sq_ja, "en": sq_en}

        er = plan.get("evidence_requirements") or {}
        if not isinstance(er, dict): er = {}
        er.setdefault("min_unique_sources", 2)
        er.setdefault("min_chunks", 3)
        er.setdefault("must_have_actor", True)
        if intent == "STRATEGIC":
            er["must_have_ttp"] = False
        else:
            if "must_have_ttp" not in er:
                er["must_have_ttp"] = ("TTP" in req.query) or ("Technique" in req.query)
        plan["evidence_requirements"] = er

        return plan

class IngestionAgent:
    """PDF解析とIOCのSpanner登録"""
    def __init__(self, db):
        self.db = db

    def process_and_register(self, file_path):
        """PDFを読み込み、Hashを抽出してSpannerのIndicatorsに登録する"""
        try:
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            logging.error(f"PDF read error: {e}")
            return f"Error reading PDF: {e}"
        
        # IOC (Hash) の抽出と登録
        hashes = set(re.findall(r"\b[A-Fa-f0-9]{32,64}\b", text))
        logging.info(f"Extracted {len(hashes)} unique hashes from {file_path}")

        if hashes and self.db:
            def _upsert(transaction):
                for h in hashes:
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO Indicators (IndicatorId, Type, Value) VALUES (@id, 'hash', @val)",
                        params={"id": _stable_uuid("indicator", f"hash:{h}"), "val": h},
                        param_types={"id": spanner.param_types.STRING, "val": spanner.param_types.STRING}
                    )
            try:
                self.db.run_in_transaction(_upsert)
            except Exception as e:
                logging.warning(f"Spanner write failed: {e}")
                return f"Text extracted but Spanner write failed: {e}"
        
        return text

class RetrievalAgent:
    """RAG + Spanner Graphのハイブリッド検索"""
    def __init__(self, db):
        self.db = db

    def _query_graph(self, hashes: List[str]) -> List[Dict[str, Any]]:
        if not hashes or not self.db:
            return []

        results: List[Dict[str, Any]] = []
        gql = """
        GRAPH CTI_Graph
        MATCH (a:Actors)-[:Conducts]->(c:Campaigns)-[:Uses]->(i:Indicators)
        WHERE i.Type = 'hash' AND i.Value = @hash
        RETURN a.Name AS actor, c.Name AS campaign, i.Value AS indicator
        """

        try:
            with self.db.snapshot() as snapshot:
                for h in hashes:
                    rows = snapshot.execute_sql(
                        gql,
                        params={"hash": h},
                        param_types={"hash": spanner.param_types.STRING},
                    )
                    for r in rows:
                        results.append({"actor": r[0], "campaign": r[1], "indicator": r[2]})
        except Exception as e:
            logging.warning(f"[GraphQuery] failed: {e}")

        return results

    def get_evidence(self, search_queries: Dict[str, str], *, top_k: int = 30, run_id: str | None = None) -> Dict[str, Any]:
        q_ja = (search_queries.get("ja") or "").strip() if isinstance(search_queries, dict) else ""
        q_en = (search_queries.get("en") or "").strip() if isinstance(search_queries, dict) else ""

        debug = {"ja": {}, "en": {}, "merged": {}}
        raw_chunks: List[Dict[str, Any]] = []

        # ja
        if q_ja:
            try:
                ch = rag_search_via_vertex(q_ja, top_n=top_k, run_id=run_id, lang="ja")
                raw_chunks.extend(ch)
                debug["ja"] = {"ok": True, "hits": len(ch)}
            except RagRetrievalError as e:
                debug["ja"] = {"ok": False, "error": str(e)}

        # en
        if q_en and q_en != q_ja:
            try:
                ch = rag_search_via_vertex(q_en, top_n=top_k, run_id=run_id, lang="en")
                raw_chunks.extend(ch)
                debug["en"] = {"ok": True, "hits": len(ch)}
            except RagRetrievalError as e:
                debug["en"] = {"ok": False, "error": str(e)}

        dedup: List[Dict[str, Any]] = []
        seen = set()
        for c in raw_chunks:
            src = (c.get("source_uri") or "unknown").strip() or "unknown"
            txt = (c.get("text") or "").strip()
            key = (src, txt[:200])
            if key in seen: continue
            seen.add(key)
            dedup.append(c)
        # 参照する文献の範囲
        chunks = _select_diverse_chunks(dedup, target_k=20, max_chars=3500, max_per_source=3)
        for i, c in enumerate(chunks, start=1): c["rank"] = i

        debug["merged"] = {"raw": len(raw_chunks), "selected": len(chunks)}

        # Graph相関
        combined_text = " ".join([c["text"] for c in chunks])
        found_hashes = re.findall(r"\b[A-Fa-f0-9]{32,64}\b", combined_text)
        graph_data = self._query_graph(found_hashes) if found_hashes else []

        return {"chunks": chunks, "graph": graph_data, "retrieval_debug": debug}

class AnalysisAgent:
    """レポート生成"""
    def generate(self, intent, reqs, bundle, feedback=""):
        # 上位20件
        chunk_text = "\n".join([f"[{c['rank']}] {c['text']} \n(source_uri={c.get('source_uri')})" for c in bundle.get('chunks', [])[:20]])
        
        prompt = f"""
            あなたはシニアCTIアナリストです。以下の情報に基づき、{intent}レベルの脅威インテリジェンスレポートを作成してください。
            
            【要件】
            - 読者: {reqs.get('audience')}
            - 期間: {reqs.get('time_range')}
            - 業界: {reqs.get('industry')}
            - 前回の修正指示: {feedback if feedback else "なし"}
            
            【RAG証拠 (信頼度 高)】
            {chunk_text}
            
            【Graph相関 (構造化データ)】
            {bundle['graph']}
            
            【ルール】
            1. 主張には必ず [1] のような形式で証拠番号を付記すること（上のRAG証拠の番号のみ使用）。
            2. 根拠のない情報・推測・一般論の水増しは禁止。根拠が足りない場合は「不明（根拠不足）」と明記する。
            3. 参照した証拠番号のないActor名、Campaign名、TTP（Txxxx）、IoCは出力しない（絶対に作り上げない）。
            4. STRATEGIC の場合、TTP（TechniqueIdの列挙）は必須ではない。根拠が薄い場合は出さない。
            5. 日本語で出力すること。
        """
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, 
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2)
            )
            return response.text
        except Exception as e:
            logging.error(f"Gemini gen failed: {e}")
            return "Error generating report."

class ReviewPolicyAgent:
    """品質チェック"""
    def check(self, markdown):
        # 簡易チェック: 引用形式 [n] が含まれているか
        has_citations = bool(re.search(r"\[\d+\]", markdown))
        passed = has_citations
        feedback = "" if passed else "【却下】すべての主張に [1] のような形式で根拠（Evidence ID）を紐付けてください。"
        return {"passed": passed, "feedback": feedback}

class GraphUpdateAgent:
    """レポートからナレッジを抽出し、グラフを更新する"""
    def __init__(self, db):
        self.db = db

    def extract_actor_campaign_artifacts(self, markdown_report: str) -> List[Dict[str, object]]:
        """
        出力例（campaign は任意）:
        [
          {
            "actor": "Mustang Panda",
            "campaign": "Operation Foo" | null,
            "iocs": [{"ioc":"1.2.3.4","type":"ip"}, {"ioc":"0a1b...","type":"hash"}],
            "techniques": [{"technique_id":"T1059","name":"...","tactic":"..."}]
          }
        ]
        """
        if not markdown_report: return []

        prompt = f"""
                あなたは脅威インテリジェンスの抽出器です。
                以下のMarkdownレポートから攻撃者名、キャンペーン名、IoC情報、TTP情報を抽出し、JSON配列で出力してください。

                    【抽出ルール】
                    1. 攻撃者（actor）が特定できない場合、actorフィールドには "Unknown Actor" またはレポート内の一般名称（例: "IoT Botnet Operator"）を入れてください。
                    2. TTP（MITRE ATT&CK Technique）は、技術ID（例：T1190）があればtechniquesフィールドとして抽出してください。
                    3. CVE番号（例: CVE-2023-1389）があれば、vulnerabilitiesフィールドとして抽出してください。そのCVE番号に関する説明はdescriptionフィールドに入れます。
                    4. 具体的名称がない概念的な記述のみの場合は含めないでください。

                    【出力JSON形式】
                    [
                    {{
                        "actor": "攻撃者名",
                        "campaign": "キャンペーン名",
                        "iocs": [{{ "ioc": "...", "type": "hash|ip|domain" }}],
                        "techniques": [{{ "technique_id": "Txxxx", "name": "..." }}],
                        "vulnerabilities": [{{ "id": "CVE-xxxx-xxxx", "description": "..." }}]
                    }}
                    ]

                Markdown:
                ---
                {markdown_report}
                ---
            """.strip()

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
            )
            return _extract_json_array(response.text)
        except Exception as e:
            logging.error(f"Extract artifacts failed: {e}")
            return []

    def update_knowledge_graph(self, markdown_report: str, run_id: Optional[str] = None) -> int:
        if not self.db: return 0
        """
        - Actor / Campaign / IoC / Technique をレポートから抽出し Spanner に反映
        - TechniqueId が無い/不正なTTPは登録しない（精度優先）
        """

        items = self.extract_actor_campaign_artifacts(markdown_report)
        if not items:
            logging.info("No {actor,campaign,iocs,techniques,vulnerabilities} extracted.")
            return 0

        fallback_campaign_name = f"Run:{run_id}" if run_id else "Run:unknown"

        def _upsert_graph(transaction):
            total_links = 0
            for it in items:
                actor_name = it.get("actor") or "Unknown Actor"
                campaign_name = it.get("campaign") or fallback_campaign_name
                actor_id = _stable_uuid("actor", actor_name)
                campaign_id = _stable_uuid("campaign", campaign_name)

                # 1) Actors（親）
                transaction.execute_update(
                    "INSERT OR IGNORE INTO Actors (ActorId, Name, Origin) VALUES (@id, @name, NULL)",
                    params={"id": actor_id, "name": actor_name},
                    param_types={"id": spanner.param_types.STRING, "name": spanner.param_types.STRING},
                )

                # 2) Campaigns
                transaction.execute_update(
                    "INSERT OR IGNORE INTO Campaigns (CampaignId, Name) VALUES (@id, @name)",
                    params={"id": campaign_id, "name": campaign_name},
                    param_types={"id": spanner.param_types.STRING, "name": spanner.param_types.STRING},
                )

                # 3) Actor -> Campaign
                transaction.execute_update(
                    "INSERT OR IGNORE INTO ActorConductsCampaign (ActorId, CampaignId, ObservedAt) VALUES (@aid, @cid, PENDING_COMMIT_TIMESTAMP())",
                    params={"aid": actor_id, "cid": campaign_id},
                    param_types={"aid": spanner.param_types.STRING, "cid": spanner.param_types.STRING},
                )

                # IoC
                for ioc_obj in (it.get("iocs") or []):
                    ioc_val = (ioc_obj.get("ioc") or "").strip()
                    ioc_type = (ioc_obj.get("type") or "other").strip().lower()
                    if not ioc_val: continue
                    ind_id = _stable_uuid("indicator", f"{ioc_type}:{ioc_val}")

                    transaction.execute_update(
                        "INSERT OR IGNORE INTO Indicators (IndicatorId, Type, Value) VALUES (@id, @type, @val)",
                        params={"id": ind_id, "type": ioc_type, "val": ioc_val},
                        param_types={"id": spanner.param_types.STRING, "type": spanner.param_types.STRING, "val": spanner.param_types.STRING},
                    )
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO CampaignUsesIndicator (CampaignId, IndicatorId, ObservedAt) VALUES (@cid, @iid, PENDING_COMMIT_TIMESTAMP())",
                        params={"cid": campaign_id, "iid": ind_id},
                        param_types={"cid": spanner.param_types.STRING, "iid": spanner.param_types.STRING},
                    )
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO ActorUsesIndicator (ActorId, IndicatorId, ObservedAt) VALUES (@aid, @iid, PENDING_COMMIT_TIMESTAMP())",
                        params={"aid": actor_id, "iid": ind_id},
                        param_types={"aid": spanner.param_types.STRING, "iid": spanner.param_types.STRING},
                    )
                    total_links += 1

                # TTP
                for t in (it.get("techniques") or []):
                    tid = (t.get("technique_id") or "").strip()
                    if not tid or not TECHNIQUE_ID_RE.match(tid): continue
                    
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO Techniques (TechniqueId, Name, Tactic) VALUES (@id, @name, @tactic)",
                        params={"id": tid, "name": t.get("name"), "tactic": t.get("tactic")},
                        param_types={"id": spanner.param_types.STRING, "name": spanner.param_types.STRING, "tactic": spanner.param_types.STRING},
                    )
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO ActorUsesTechnique (ActorId, TechniqueId, ObservedAt) VALUES (@aid, @tid, PENDING_COMMIT_TIMESTAMP())",
                        params={"aid": actor_id, "tid": tid},
                        param_types={"aid": spanner.param_types.STRING, "tid": spanner.param_types.STRING},
                    )
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO CampaignUsesTechnique (CampaignId, TechniqueId, ObservedAt) VALUES (@cid, @tid, PENDING_COMMIT_TIMESTAMP())",
                        params={"cid": campaign_id, "tid": tid},
                        param_types={"cid": spanner.param_types.STRING, "tid": spanner.param_types.STRING},
                    )
                    total_links += 1
                    
                # Vulnerability
                for v in (it.get("vulnerabilities") or []):
                    v_id = v.get("id", "").strip().upper()
                    if not v_id.startswith("CVE-"): continue

                    # 1. Vulnerabilities マスター登録
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO Vulnerabilities (VulnerabilityId, Description) VALUES (@id, @desc)",
                        params={"id": v_id, "desc": v.get("description")},
                        param_types={"id": spanner.param_types.STRING, "desc": spanner.param_types.STRING},
                    )

                    # 2. Campaign -> Vulnerability の紐付け
                    transaction.execute_update(
                        "INSERT OR IGNORE INTO CampaignExploitsVulnerability (CampaignId, VulnerabilityId, ObservedAt) VALUES (@cid, @vid, PENDING_COMMIT_TIMESTAMP())",
                        params={"cid": campaign_id, "vid": v_id},
                        param_types={"cid": spanner.param_types.STRING, "vid": spanner.param_types.STRING},
                    )

                    # 3. Technique -> Vulnerability の紐付け (もし手法とCVEが文脈的に近い場合)
                    for t in (it.get("techniques") or []):
                        tid = (t.get("technique_id") or "").strip()
                        if not tid: continue
                        transaction.execute_update(
                            "INSERT OR IGNORE INTO TechniqueExploitsVulnerability (TechniqueId, VulnerabilityId, ObservedAt) VALUES (@tid, @vid, PENDING_COMMIT_TIMESTAMP())",
                            params={"tid": tid, "vid": v_id},
                            param_types={"tid": spanner.param_types.STRING, "vid": spanner.param_types.STRING},
                        )
                    total_links += 1

            return total_links

        try:
            total_created = self.db.run_in_transaction(_upsert_graph)
            logging.info(f"Knowledge Graph updated. Total links: {total_created}")
            return total_created
        except Exception as e:
            logging.error(f"Graph upsert failed: {e}")
            return 0

    # intel_outputsテーブルへの保存メソッド
    def save_run_log(self, run_id: str, intent: str, content: str):
        if not self.db: return
        try:
            def _insert_log(tx):
                tx.execute_update(
                    """
                    INSERT INTO intel_outputs (run_id, intent_level, content, created_at)
                    VALUES (@run_id, @intent, @content, PENDING_COMMIT_TIMESTAMP())
                    """,
                    params={"run_id": run_id, "intent": intent, "content": content},
                    param_types={
                        "run_id": spanner.param_types.STRING,
                        "intent": spanner.param_types.STRING,
                        "content": spanner.param_types.STRING
                    }
                )
            self.db.run_in_transaction(_insert_log)
            logging.info(f"Saved run log to intel_outputs for run_id={run_id}")
        except Exception as e:
            logging.error(f"Failed to save run log to Spanner: {e}")

class SupervisorAgent:
    """全体の調整役"""
    def __init__(self, req: AgentRunRequest):
        self.run_id = str(uuid.uuid4())
        self.req = req
        self.ingestion = IngestionAgent(spanner_db)
        self.retrieval = RetrievalAgent(spanner_db)
        self.planner = QueryPlannerAgent()
        self.analysis = AnalysisAgent()
        self.review = ReviewPolicyAgent()
        self.graph_update = GraphUpdateAgent(spanner_db)

    def run(self):
        logging.info(f"Starting run {self.run_id}")
        
        # 1) 検索計画（intent判定含む）: LLMで決める
        plan = self.planner.plan(self.req)
        intent = plan.get("intent_level") or "TACTICAL"

        # 2) 証拠収集: LLMが作った search_queries（ja/en）を使う / topK=30
        sq = plan.get("search_queries") or {"ja": self.req.query, "en": ""}
        try:
            bundle = self.retrieval.get_evidence(sq, top_k=30, run_id=self.run_id)
        except RagRetrievalError as e:
            return f"# RAG取得失敗\nError: {e}", intent, False

        # 3) 根拠ゲート: 足りなければ生成しない
        gate = _evidence_gate(plan, bundle.get("chunks") or [])
        if not gate.get("ok"):
            markdown = _insufficient_report(self.req.dict(), plan, gate, bundle.get("chunks") or [])
#            self.graph_update.save_run_log(self.run_id, intent, markdown)
            return markdown, intent, True
        
        # 4) 生成とレビューのループ (最大2回)
        last_feedback = ""
        final_markdown = ""
        success = False
        
        for i in range(3):
            final_markdown = self.analysis.generate(intent, self.req.dict(), bundle, last_feedback)
            review_result = self.review.check(final_markdown)
            if review_result["passed"]:
                success = True
                try:
                    # [ADD] Spannerへの結果保存
                    self.graph_update.save_run_log(self.run_id, intent, final_markdown)
                    # Graph更新
                    self.graph_update.update_knowledge_graph(final_markdown, run_id=self.run_id)
                except Exception as e:
                    logging.warning(f"Graph update/save error: {e}")
                break
            last_feedback = review_result["feedback"]

        if not success:
             self.graph_update.save_run_log(self.run_id, intent, final_markdown)

        return final_markdown, intent, success

# ---------------------------------------------------------
# 5. Endpoints
# ---------------------------------------------------------
@app.post("/agent_run", response_model=AgentRunResponse)
def agent_run(req: AgentRunRequest):
    supervisor = SupervisorAgent(req)
    markdown, intent, success = supervisor.run()
    if success:
        save_to_gcs(supervisor.run_id, markdown)

    return AgentRunResponse(
        run_id=supervisor.run_id,
        intent_level=intent,
        markdown=markdown,
        review_status=success,
        feedback=None if success else "品質基準を満たせずリトライ上限に達しました。"
    )

@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    PDFをアップロードし、テキスト抽出してSpannerのIndicatorsに登録する。
    RAGへの登録は済んでいる前提だが、Spanner GraphへのIoCシード投入として利用可能。
    """
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ingest_agent = IngestionAgent(spanner_db)
    
    # 簡易的に同期実行（重い場合はBackgroundTasks推奨）
    result_text = ingest_agent.process_and_register(file_path)
    os.remove(file_path)
    
    return {"filename": file.filename, "status": "processed", "preview": result_text[:200]}

@app.post("/update_graph")
def update_graph(req: UpdateGraphRequest):
    """
    指定されたGCS上のレポートからActor, IoC, TTPを抽出し、Spanner Graphを更新する。
    """
    # 1. GCSからレポート内容を取得
    report_content = load_from_gcs(req.bucket_name, req.file_path)
    
    # 2. GraphUpdateAgentを使用して抽出と更新を実行
    update_agent = GraphUpdateAgent(spanner_db)
    
    # 抽出とDB登録の実行
    total_links = update_agent.update_knowledge_graph(report_content, run_id=req.file_path)
    
    return {
        "status": "success",
        "file_processed": req.file_path,
        "links_created": total_links
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
