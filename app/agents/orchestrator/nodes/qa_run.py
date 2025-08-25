from __future__ import annotations

import logging

from app.agents.orchestrator.state import AgentState
from app.agents.orchestrator.tools import (
    run_t2s_agent_with_instruction,
    visualize_and_explain,
)
from app.agents.orchestrator.tools.web_executor import execute_web_from_plan
from app.agents.visualizer.graph import build_visualize_graph
from app.agents.visualizer.state import VisualizeState

logger = logging.getLogger(__name__)

def qa_run_node(state: AgentState) -> AgentState:
    logger.info("===== 📝 QA 러너 노드 실행 =====")

    plan = state.qa_plan
    sql_context = state.sql_context

    if not plan or not sql_context:
        logger.error("===== 📝 QA 러너 노드 실행 실패 =====")
        return state
    
    if plan.mode == "pass": 
        instruction = state.user_message
    else: 
        instruction = f"{state.user_message}\n\n보충 요청: {plan.augment.strip()}" if plan.augment else state.user_message
    
    if plan.use_t2s:
        state.qa_table = run_t2s_agent_with_instruction(sql_context, instruction)
        if plan.visualize: 
            viz = visualize_and_explain(
                question=state.user_message or "",
                data_rows=state.qa_table.get("rows", []) if state.qa_table else [],
                instruction=plan.viz_hint or "",
                model="gemini-2.5-flash",
                graph_builder=build_visualize_graph,
                state_cls=VisualizeState,
            )
            state.qa_chart = viz["json_graph"]

    if plan.use_web:
        source = plan.web_source 
        state.qa_web_row = execute_web_from_plan(source, state.user_message)

    return state
