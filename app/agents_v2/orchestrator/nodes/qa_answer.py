# agent/nodes/qa/qa_build_answer_node.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.agents_v2.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

# ---- Prompt ----
ANSWER_PROMPT = """
당신은 마케팅 데이터/트렌드 Q&A 도우미입니다.
아래 입력을 바탕으로 한국어로 정확하고 실무 친화적으로 답하세요.

[사용자 질문]
{question}

[내부 표(미리보기 최대 10행)]
{table_preview}

[외부 트렌드/참고 메모]
{snapshot_notes}

규칙:
1) 표가 존재하면 표 기반 핵심 결론 2~3개를 한 문장씩 요약합니다.
2) 스냅샷 메모가 있으면 보강 정보로 1~2문장 추가합니다. 단, 표와 모순될 경우 표를 우선합니다.
3) 과도한 확정 표현 대신 '경향으로 보입니다' 수준의 표현을 사용합니다.
4) 3~6문장 내로 끝냅니다. 불필요한 장황함 금지.
"""

# ---- Utilities ----
def _preview_rows(table: Dict[str, Any], n: int = 10) -> str:
    try:
        rows = (table or {}).get("rows") or []
        head = rows[:n]
        return json.dumps(head, ensure_ascii=False)
    except Exception:
        return "[]"

def _sources_from_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, str]]:
    if not isinstance(snapshot, dict):
        return []
    out = []
    for s in (snapshot.get("sources") or []):
        title = (s or {}).get("title") or ""
        url = (s or {}).get("url") or ""
        if title or url:
            out.append({"title": title, "url": url})
    return out

# ---- Node ----
def qa_build_answer_node(state: AgentState) -> AgentState:
    """
    입력 (있을 수 있는 키):
      - user_message: str
      - qa_table: {rows, columns, row_count}
      - qa_chart: str (Plotly JSON)
      - qa_snapshot: {notes: [..], sources: [...]}

    출력:
      state["response"] = {
        "text": str,                         # 최종 답변
        "attachments": {
          "chart_json": str|None,            # Plotly Figure JSON
          "table_preview": List[dict]|None,  # rows 상위 10개
          "sources": List[{title,url}]       # 외부 참고 링크
        },
        "await_user": False
      }
    """
    question: str = state.user_message or ""
    table: Dict[str, Any] = state.qa_table or {}
    snapshot: Dict[str, Any] = state.qa_snapshot or {}
    chart_json: Optional[str] = state.qa_chart

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=settings.GOOGLE_API_KEY,
        )
        prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
        llm_text = llm.invoke({
            "question": question,
            "table_preview": _preview_rows(table, n=10),
            "snapshot_notes": "; ".join(snapshot.get("notes") or []),
        }).content.strip()
    except Exception as e:
        logger.exception("[qa_build_answer_node] LLM 요약 실패: %s", e)
        llm_text = ""

    parts: List[str] = []
    if llm_text:
        parts.append(llm_text)
    final_text = "\n\n".join(parts) if parts else "요청하신 내용을 바탕으로 핵심을 요약했습니다."

    return {"response": final_text, "graph": chart_json, "table": (table.get("rows") or [])[:10], "snapshot": snapshot}
