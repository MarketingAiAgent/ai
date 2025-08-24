import logging 

from app.agents_v2.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

def ask_scope_period_node(state: AgentState):
    logger.info("===== 🚀 브랜드/제품 기준 질문 노드 실행 =====")
    missing = [x for x in ["scope", "period"] if state.promotion_slots.get(x) is None]
    if missing == ["scope"]:
        return {"response": "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시겠습니까?"}
    elif missing == ["period"]:
        return {"response": "기간은 어느 정도로 계획하고 계신가요?"}
    else:
        return {"response": "브랜드 기준과 제품 기준 중 어느 쪽으로 진행하시고, 기간은 어느 정도로 보실까요?"}