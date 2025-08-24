from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    options_generator_node as _orig_options_generator_node,
)

def options_generator_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_options_generator_node(state)
