from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    visualizer_caller_node as _orig_visualizer_caller_node,
)

def visualizer_caller_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_visualizer_caller_node(state)
