
"""
v2 라우터 모듈.
내부 로직은 기존 orchestrator.graph 내 라우터 함수를 그대로 호출합니다.
이렇게 하면 로직 변경 없이 구조만 개선할 수 있습니다.
"""
from app.agents.orchestrator.state import OrchestratorState

# 원본 라우터 함수 import
from app.agents.orchestrator.graph import (  # type: ignore
    _planner_router as _orig_planner_router,
    _action_router as _orig_action_router,
    _should_visualize_router as _orig_should_visualize_router,
)

def planner_router(state: OrchestratorState) -> str:
    return _orig_planner_router(state)

def action_router(state: OrchestratorState) -> str:
    return _orig_action_router(state)

def should_visualize_router(state: OrchestratorState) -> str:
    return _orig_should_visualize_router(state)
