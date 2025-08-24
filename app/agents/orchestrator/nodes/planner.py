from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    planner_node as _orig_planner_node,
)

def planner_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_planner_node(state)
