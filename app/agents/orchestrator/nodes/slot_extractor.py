from app.agents.orchestrator.state import OrchestratorState
from typing import Dict, Any
from app.agents.orchestrator.graph import (
    slot_extractor_node as _orig_slot_extractor_node,
)

def slot_extractor_node(state: OrchestratorState) -> Dict[str, Any]:
    return _orig_slot_extractor_node(state)
