from __future__ import annotations

import logging 
import json 
from typing import List, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from app.core.config import settings


logger = logging.getLogger(__name__)

from app.agents_v2.orchestrator.state import AgentState, PromotionSlots 

# ===== prompt =====
def _build_messages(history: List[str], current: str) -> List[tuple]:
    # 직전 4개까지만 사용
    history = (history or [])[-4:]

    system = (
        "당신은 한국어로 동작하는 정보 추출기입니다. "
        "입력으로 이전 AI 질문과 현재 유저 응답을 보고, 다음 필드만 JSON으로 추출해 반환하세요:\n"
        "{{\n"
        '  "audience": string|null,\n'
        '  "scope": "브랜드"|"제품"|null,\n'
        '  "target": string|null,\n'
        '  "period": string|null,\n'
        '  "KPI": string|null,\n'
        '  "concept": string|null\n'
        "}}\n\n"
        "규칙:\n"
        "1) 원문에 명시되지 않은 값은 추정하지 말고 null로 두세요.\n"
        "2) 기간은 간단 표현 유지(예: '다음 달 4주').\n"
        "3) 설명/주석 없이 반드시 JSON만 출력하세요."
    )

    fewshot = [
        '''
history: [{{"speaker": "user", "content": "20대 대상 프로모션 해볼래!"}}, {{"speaker": "ai", "content": "좋습니다. **브랜드 기준**과 **제품 기준** 중 어느 쪽으로 진행하시겠습니까?"}}]
user_message: "브랜드로 진행해줘" 
output: {{"audience": "20대", "scope": "브랜드", "target": null, "period": null, "KPI": null, "concept": null}}
''',
'''
history: [{{"speaker": "user", "content": "브랜드로 가자"}}, {{"speaker": "ai", "content": "기간은 어떻게 하시겠습니까? 다음 달 4주로 진행하시겠습니까?"}}]
user_message: "다음 달 4주로"
output: {{"audience": null, "scope": "브랜드", "target": null, "period": "다음 달 4주", "KPI": null, "concept": null}}
'''
]

    user = ("human", json.dumps({"history": history, "user_message": current}, ensure_ascii=False))
    return [("system", system), *fewshot, user]

# ===== pydantic =====
class SlotExtractorOutput(BaseModel):
    audience: Optional[str] = None
    scope: Optional[str] = None
    target: Optional[str] = None
    period: Optional[str] = None
    KPI: Optional[str] = None
    concept: Optional[str] = None

# ===== Node =====
def slot_extractor_node(state: AgentState):
    logger.info("===== 🧩 슬롯 추출 노드 실행 =====")
    messages = _build_messages(state.history, state.user_message)
    prompt = ChatPromptTemplate.from_messages(messages)
    parser = PydanticOutputParser(pydantic_object=SlotExtractorOutput)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=settings.GOOGLE_API_KEY
    )
    try:
        result: SlotExtractorOutput = (prompt | llm | parser).invoke()
        logger.info(f"결과: {result}")
        logger.info(f"===== ❓ 슬롯 추출 노드 실행 완료 =====")
        promotion_slots = state.promotion_slots.merge_missing(result.model_dump())
        return {"promotion_slots": promotion_slots}
    except Exception as e:
        logger.error(f"===== ❓ 슬롯 추출 노드 실행 중 오류 발생 =====")
        logger.error(f"오류 내용: {e}")
        return {"promotion_slots": None}