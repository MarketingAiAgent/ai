from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    tool_executor_node as _orig_tool_executor_node,
)

def tool_executor_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_tool_executor_node(state)
