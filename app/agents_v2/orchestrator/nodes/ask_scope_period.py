import logging 

from app.agents_v2.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

def ask_scope_period_node(state: AgentState) -> AgentState:
    logger.info("===== 🚀 브랜드/제품 기준 질문 노드 실행 =====")
    
    if not state.promotion_slots:
        return state.model_copy(update={
            "response": "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시고, 기간은 어느 정도로 보실까요?",
            "expect_fields": ["scope", "period"]
        })
    
    missing = []
    if state.promotion_slots.scope is None:
        missing.append("scope")
    if state.promotion_slots.period is None:
        missing.append("period")
    
    if missing == ["scope"]:
        return state.model_copy(update={
            "response": "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시겠습니까?",
            "expect_fields": ["scope"]
        })
    elif missing == ["period"]:
        return state.model_copy(update={
            "response": "기간은 어느 정도로 계획하고 계신가요?",
            "expect_fields": ["period"]
        })
    else:
        return state.model_copy(update={
            "response": "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시고, 기간은 어느 정도로 보실까요?",
            "expect_fields": ["scope", "period"]
        })