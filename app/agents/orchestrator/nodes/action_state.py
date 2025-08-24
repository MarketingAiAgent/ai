from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    action_state_node as _orig_action_state_node,
)

def action_state_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_action_state_node(state)
