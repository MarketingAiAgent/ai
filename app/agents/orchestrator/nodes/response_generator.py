from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    response_generator_node as _orig_response_generator_node,
)

def response_generator_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_response_generator_node(state)
