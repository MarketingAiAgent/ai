import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from app.config import settings
from typing import List, Dict, Literal
import json
from app.agents_v2.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

# ===== prompt =====
def _build_messages(history: List[Dict[str, str]], current: str) -> List[Dict[str, str]]:
        system = (
            "당신은 한국어로 동작하는 라우팅 분류기입니다. "
            "입력으로 직전 대화 히스토리와 현재 질문을 받고, "
            "다음 중 하나만 JSON으로 반환하세요:\n"
            '{ "intent": "Q&A" | "Promotion" | "Irrelevance" }\n\n'
            "- 'Q&A': 지표/데이터/트렌드/지식 질의(예: CTR 알려줘, 트렌드 요약 등)\n"
            "- 'Promotion': 프로모션/캠페인/이벤트 기획 의도(예: 20대 대상 프로모션 하자, 브랜드/제품 기준 등)\n"
            "- 'Out-of-scope': 인사, 잡담, 시스템 명령 등 비관련 요청\n"
            "설명이나 여분 텍스트 없이 반드시 JSON 한 줄만 출력하세요."
        )

        fewshot = [
            {
                "role": "user",
                "content": json.dumps({
                    "history": ["지난달 CRM 성과 어땠지?"],
                    "current": "CTR이랑 전환율 알려줘"
                }, ensure_ascii=False)
            },
            {"role": "assistant", "content": json.dumps({"intent": "Q&A"}, ensure_ascii=False)},
            {
                "role": "user",
                "content": json.dumps({
                    "history": ["봄 시즌 캠페인 준비할까?"],
                    "current": "20대 대상 프로모션 해볼래!"
                }, ensure_ascii=False)
            },
            {"role": "assistant", "content": json.dumps({"intent": "Promotion"}, ensure_ascii=False)},
        ]

        user = {
            "role": "user",
            "content": json.dumps({"history": history, "current": current}, ensure_ascii=False),
        }

        return [{"role": "system", "content": system}, *fewshot, user]

# ===== pydantic =====
class RouterOutput(BaseModel):
    intent: Literal["QA", "Promotion", "Out-of-scope"] = Field(description="유저의 의도 파악하여 다음 단계 진행")

# ===== Node =====
def router_node(state: AgentState):
    logger.info("===== 🤔 라우터 수립 노드 실행 =====")
    messages = _build_messages(state.history, state.user_message)
    prompt = ChatPromptTemplate.from_messages(messages)
    parser = PydanticOutputParser(pydantic_object=RouterOutput)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        api_key=settings.GOOGLE_API_KEY
    )

    try: 
        result: RouterOutput = (prompt | llm | parser).invoke()
        logger.info(f"결과: {result.intent}")
        logger.info(f"===== 🤔 라우터 수립 노드 실행 완료 =====")
        return {"intent": result.intent}

    except Exception as e:
        logger.error(f"===== 🤔 라우터 수립 노드 실행 중 오류 발생 =====")
        logger.error(f"오류 내용: {e}")
        return {"intent": "Out-of-scope"}
