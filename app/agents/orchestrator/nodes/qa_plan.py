from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Literal, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agents.orchestrator.state import AgentState, QAPlan
from app.core.config import settings

logger = logging.getLogger(__name__)


# -------------------- Prompt --------------------

def _build_messages(history: List[str], question: str) -> List[Tuple[str,str]]:
    system = (
        "너는 마케팅 Q&A 라우터다. 아래 JSON을 출력하라 (설명/여분 금지).\n"
        "{\n"
        '  "mode": "pass" | "augment",\n'
        '  "augment": "질문이 모호할 때 최소 보충 한 줄(한국어)" | null,\n'
        '  "use_t2s": true | false,\n'
        '  "use_web": true | false,\n'
        '  "web_source": Literal["supabase_marketing" | "supabase_beauty" | "tavily"],\n'
        '  "need_visual": true | false\n'
        "}\n"
        "규칙:\n"
        "- 질문이 내부 수치/지표/추이면 use_t2s=true, use_web=false.\n"
        "- 업계/사례/트렌드/키워드/인플루언서/리포트 등은 use_t2s=false, use_web=true.\n"
        "- 유저가 추세를 보여달라는 등의 형식으로 질문을 하여 시각화 필요할 경우 need_visual=true 아니라면 false.\n"
        "- 외부 지식 필요로 하여 use_web=true 일 때 web_source에 어떤 소스로 정보를 얻을지 추가하라. (예: ['supabase_marketing', 'supabase_beauty', 'tavily'])\n"
        "- 질문이 충분히 구체적이면 mode=pass, 애매하면 mode=augment로 한 줄만 보충(예: 기간, 그레인, 세분화).\n"
        "- 스키마/테이블/컬럼명 언급 금지. 자연어만."
    )
    fewshot = [
        ("human", json.dumps({"history":["채널별로도 봤고"], "question":"지난 1년간 연령대별 방문수 추이 보여줘"}, ensure_ascii=False)),
        ("ai", json.dumps({"mode":"pass","augment":None,"use_t2s":True,"use_web":False,"need_visual":True}, ensure_ascii=False)),
        ("human", json.dumps({"history":["신규 유입이 줄었어"], "question":"채널 성과 좀 보여줘"}, ensure_ascii=False)),
        ("ai", json.dumps({"mode":"augment","augment":"최근 30일 기준, 채널별 핵심 지표만 간단 비교해줘","use_t2s":True,"use_web":False,"need_visual":False}, ensure_ascii=False)),
        ("human", json.dumps({"history":["여름 캠페인 준비"], "question":"뷰티 인플루언서 트렌드 핵심 키워드 알려줘"}, ensure_ascii=False)),
        ("ai", json.dumps({"mode":"pass","augment":None,"use_t2s":False,"use_web":True,"need_visual":False}, ensure_ascii=False)),
    ]
    user = ("human", json.dumps({"history": history[-2:], "question": question}, ensure_ascii=False))
    return [("system", system), *fewshot, user]


# -------------------- Node --------------------

def qa_plan_node(state: AgentState) -> AgentState:
    logger.info("===== 📝 QA 플래너 노드 실행 =====")

    history = state.history or []
    question = state.user_message or ""

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
    parser = PydanticOutputParser(pydantic_object=QAPlan)

    try:
        msgs = _build_messages(history, question)
        out: QAPlan = (llm | parser).invoke(msgs)
        logger.info(f"QA 플래너 노드 실행 결과: {out}")
        logger.info(f"===== 📝 QA 플래너 노드 실행 완료 =====")
        return {"qa_plan": out}
    except Exception:
        logger.exception("[qa_route_node] 실패 → pass-through 기본")
        out = QAPlan(mode="pass", augment=None, use_t2s=True, use_web=False, need_visual=True)
        return {"qa_plan": out}