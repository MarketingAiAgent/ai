# orchestrator/tools/visualize_executor.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def visualize_and_explain(
    *,
    question: str,
    data_rows: List[Dict[str, Any]],
    instruction: str = "",
    model: str = "gemini-2.5-flash",
    graph_builder=None,
    state_cls=None,
) -> Dict[str, Any]:
    
    if graph_builder is None or state_cls is None:
        logger.error("visualize_and_explain: graph_builder와 state_cls를 주입하세요.")
        raise RuntimeError("visualize_and_explain: graph_builder와 state_cls를 주입하세요.")

    app = graph_builder(model)
    st = state_cls(
        user_question=question,
        json_data=json.dumps(data_rows, ensure_ascii=False),
        instruction=instruction or "",
    )

    out = app.invoke(st)

    if hasattr(out, "json_graph"):
        json_graph = out.json_graph
    elif isinstance(out, dict):
        json_graph = out.get("json_graph")
    else:
        json_graph = None

    if not json_graph:
        logger.error("visualize_and_explain: json_graph가 비었습니다.")
        raise RuntimeError("visualize_and_explain: json_graph가 비었습니다.")

    return {"json_graph": json_graph}
