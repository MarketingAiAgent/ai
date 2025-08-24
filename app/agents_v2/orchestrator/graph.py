from langgraph.graph import StateGraph, START, END

from app.agents_v2.orchestrator.nodes.router import router_node
from app.agents_v2.orchestrator.nodes.slot_extractor import slot_extractor_node
from app.agents_v2.orchestrator.nodes.ask_scope_period import ask_scope_period_node
from app.agents_v2.orchestrator.nodes.qa_plan import qa_plan_node
from app.agents_v2.orchestrator.nodes.qa_run import qa_run_node
from app.agents_v2.orchestrator.nodes.qa_answer import qa_build_answer_node
from app.agents_v2.orchestrator.nodes.promotion_report import generate_promotion_report_node
from app.agents_v2.orchestrator.nodes.plan_options import plan_option_prompts_node
from app.agents_v2.orchestrator.nodes.ask_target import build_options_and_question_node
from app.agents_v2.orchestrator.nodes.out_of_scope import out_of_scope_node
from app.agents_v2.orchestrator.state import AgentState

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)

workflow.add_node("qa_plan", qa_plan_node)
workflow.add_node("qa_run", qa_run_node)
workflow.add_node("qa_answer", qa_build_answer_node)

workflow.add_node("slot_extractor", slot_extractor_node)
workflow.add_node("ask_scope_period", ask_scope_period_node)
workflow.add_node("plan_options", plan_option_prompts_node)
workflow.add_node("ask_target", build_options_and_question_node)
workflow.add_node("promotion_report", generate_promotion_report_node)

workflow.add_node("out_of_scope", out_of_scope_node)