from langgraph.graph import StateGraph, START, END

from app.agents.orchestrator.nodes.router import router_node
from app.agents.orchestrator.nodes.slot_extractor import slot_extractor_node
from app.agents.orchestrator.nodes.ask_scope_period import ask_scope_period_node
from app.agents.orchestrator.nodes.qa_plan import qa_plan_node
from app.agents.orchestrator.nodes.qa_run import qa_run_node
from app.agents.orchestrator.nodes.qa_answer import qa_build_answer_node
from app.agents.orchestrator.nodes.promotion_report import generate_promotion_report_node
from app.agents.orchestrator.nodes.plan_options import plan_option_prompts_node
from app.agents.orchestrator.nodes.ask_target import build_options_and_question_node
from app.agents.orchestrator.nodes.out_of_scope import out_of_scope_node
from app.agents.orchestrator.state import AgentState

# 분기 함수들
def route_by_intent(state: AgentState) -> str:
    """라우터 결과에 따라 분기"""
    intent = state.intent
    if intent == "QA":
        return "qa_plan"
    elif intent == "Promotion":
        return "slot_extractor"
    else:
        return "out_of_scope"

def route_promotion_flow(state: AgentState) -> str:
    """프로모션 플로우 분기"""
    if not state.promotion_slots:
        return "ask_scope_period"
    
    action, _ = state.promotion_slots.decide_next_action()
    
    if action == "ASK_SCOPE_PERIOD":
        return "ask_scope_period"
    elif action == "ASK_TARGET_WITH_OPTIONS":
        return "plan_options"
    elif action == "RECAP_CONFIRM":
        return "promotion_report"
    else:
        return "ask_scope_period"

def route_after_plan_options(state: AgentState) -> str:
    """옵션 계획 후 분기"""
    if not state.promotion_slots:
        return "ask_scope_period"
    
    action, _ = state.promotion_slots.decide_next_action()
    
    if action == "ASK_TARGET_WITH_OPTIONS":
        return "ask_target"
    else:
        return "ask_scope_period"

def route_after_ask_target(state: AgentState) -> str:
    """타겟 질문 후 분기"""
    if not state.promotion_slots:
        return "ask_scope_period"
    
    action, _ = state.promotion_slots.decide_next_action()
    
    if action == "RECAP_CONFIRM":
        return "promotion_report"
    else:
        return "ask_target"

def route_after_qa_plan(state: AgentState) -> str:
    """QA 계획 후 분기"""
    if state.qa_plan.use_t2s or state.qa_plan.use_web:
        return "qa_run"
    else:
        return "qa_answer"
        
# 그래프 구성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("router", router_node)

# QA 플로우
workflow.add_node("qa_plan", qa_plan_node)
workflow.add_node("qa_run", qa_run_node)
workflow.add_node("qa_answer", qa_build_answer_node)

# 프로모션 플로우
workflow.add_node("slot_extractor", slot_extractor_node)
workflow.add_node("ask_scope_period", ask_scope_period_node)
workflow.add_node("plan_options", plan_option_prompts_node)
workflow.add_node("ask_target", build_options_and_question_node)
workflow.add_node("promotion_report", generate_promotion_report_node)

# 기타
workflow.add_node("out_of_scope", out_of_scope_node)

# 엔트리 포인트 설정
workflow.set_entry_point("router")

# 엣지 추가
workflow.add_conditional_edges("router", route_by_intent, {
    "qa_plan": "qa_plan",
    "slot_extractor": "slot_extractor",
    "out_of_scope": "out_of_scope"
})

# QA 플로우
workflow.add_conditional_edges("qa_plan", route_after_qa_plan, {
    "qa_run": "qa_run",
    "qa_answer": "qa_answer"
})
workflow.add_edge("qa_run", "qa_answer")
workflow.add_edge("qa_answer", END)

# 프로모션 플로우
workflow.add_conditional_edges("slot_extractor", route_promotion_flow, {
    "ask_scope_period": "ask_scope_period",
    "plan_options": "plan_options",
    "promotion_report": "promotion_report"
})

workflow.add_conditional_edges("ask_scope_period", route_promotion_flow, {
    "ask_scope_period": "ask_scope_period",
    "plan_options": "plan_options",
    "promotion_report": "promotion_report"
})

workflow.add_conditional_edges("plan_options", route_after_plan_options, {
    "ask_target": "ask_target",
    "ask_scope_period": "ask_scope_period"
})

workflow.add_conditional_edges("ask_target", route_after_ask_target, {
    "promotion_report": "promotion_report",
    "ask_target": "ask_target"
})

workflow.add_edge("promotion_report", END)
workflow.add_edge("out_of_scope", END)

# 그래프 컴파일
orchestrator_app = workflow.compile()

__all__ = ["workflow", "orchestrator_app"]