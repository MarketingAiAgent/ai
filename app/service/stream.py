from app.agents_v2.orchestrator.state import AgentState
from app.agents_v2.orchestrator.graph import workflow
from typing import Dict, Any 

NODE_DISPLAY_MAP = {
    "route_intent": "유저 의도 파악하는 중",
    "promo_extract_slots": "프로모션 진행 상황 업데이트 중",
    "promo_ask_scope_period": "응답 생성 중",
    "promo_plan_options": "선택지 생성 중",
    "promo_run_tools": "선택지 생성을 위해 조사 중",
    "promo_build_options_and_question": "응답 생성 중",
    "promo_generate_report": "프로모션 리포트 생성 중",
    "qa_plan": "유저 질문 답변 계획 생성 중",
    "qa_run": "유조 질문 답변 위해 정보를 찾는 중",
    "qa_answer": "응답 생성 중",
    "irrelevance": "응답 생성 중",
}


TOOL_DISPLAY_MAP = {
    "sql": "내부 데이터베이스 조회 중",
    "web": "웹/트렌드 검색 중",
}

def _sse(event: Dict[str, Any]) -> str:
    """SSE 라인 직렬화"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

