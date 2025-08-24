import logging

from app.agents_v2.orchestrator.state import AgentState

logger = logging.getLogger(__name__)

def out_of_scope_node(state: AgentState):
    logger.info("===== ❌ 아웃 오브 스코프 노드 실행 =====")
    return {"response": "죄송합니다. 현재 지원하지 않는 기능입니다."}