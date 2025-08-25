import logging 

from app.agents.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

def ask_scope_period_node(state: AgentState) -> AgentState:
    logger.info("===== 🚀 브랜드/제품 기준 질문 노드 실행 =====")
    
    if not state.promotion_slots:
        return state.model_copy(update={
            "response": "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시고, 기간은 어느 정도로 보실까요?",
            "expect_fields": ["scope", "period"]
        })
    
    action, expect_fields = state.promotion_slots.decide_next_action()
    
    if action == "ASK_SCOPE_PERIOD":
        if "scope" in expect_fields and "period" in expect_fields:
            response = "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시고, 기간은 어느 정도로 보실까요?"
        elif "scope" in expect_fields:
            response = "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시겠습니까?"
        elif "period" in expect_fields:
            response = "기간은 어느 정도로 계획하고 계신가요?"
        else:
            response = "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시고, 기간은 어느 정도로 보실까요?"
    else:
        # 다른 액션으로 분기해야 하는 경우
        response = "다음 단계로 진행하겠습니다."
    
    return state.model_copy(update={
        "response": response,
        "expect_fields": expect_fields
    })