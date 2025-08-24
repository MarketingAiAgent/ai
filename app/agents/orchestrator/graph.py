
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agents.orchestrator.state import OrchestratorState  # 기존 상태 모델 재사용

# 라우터/노드: v2 구조에서 import. 실제 구현은 기존 orchestrator 모듈의 함수를 감싼 thin wrapper입니다.
from .routers import planner_router as _planner_router, action_router as _action_router, should_visualize_router as _should_visualize_router

from .nodes.planner import planner_node
from .nodes.slot_extractor import slot_extractor_node
from .nodes.action_state import action_state_node
from .nodes.options import options_generator_node
from .nodes.tool_executor import tool_executor_node
from .nodes.visualizer_bridge import visualizer_caller_node
from .nodes.response_generator import response_generator_node


# ===== Graph Assembly (조립만) =====
workflow = StateGraph(OrchestratorState)

workflow.add_node("planner", planner_node)
workflow.add_node("slot_extractor", slot_extractor_node)
workflow.add_node("action_state", action_state_node)
workflow.add_node("options_generator", options_generator_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("visualizer", visualizer_caller_node)
workflow.add_node("response_generator", response_generator_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    _planner_router,
    {
        "slot_extractor": "slot_extractor",
        "tool_executor": "tool_executor",
        "response_generator": "response_generator",
    },
)

workflow.add_edge("slot_extractor", "action_state")
workflow.add_conditional_edges(
    "action_state",
    _action_router,
    {
        "options_generator": "options_generator",
        "response_generator": "response_generator",
    },
)

workflow.add_conditional_edges(
    "tool_executor",
    _should_visualize_router,
    {
        "visualize": "visualizer",
        "skip_visualize": "response_generator",
    },
)

workflow.add_edge("options_generator", "response_generator")
workflow.add_edge("visualizer", "response_generator")
workflow.add_edge("response_generator", END)

orchestrator_app = workflow.compile()
