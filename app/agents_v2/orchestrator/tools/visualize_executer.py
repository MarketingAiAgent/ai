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
    """
    기존 시각화 그래프를 실행하는 얇은 래퍼.

    Args:
        question: 사용자의 원 질문
        data_rows: 테이블 데이터 (list of dict). 상위 N행은 upstream에서 잘라서 전달 가능
        instruction: (선택) 플래너가 지정한 시각화 힌트
        model: 사용 모델명 (그래프 빌더로 전달)
        graph_builder: build_visualize_graph(model) -> app (주입 필수)
        state_cls: VisualizeState 클래스 (user_question/json_data/instruction 필드 필요)

    Returns:
        {"json_graph": str, "explanation": str}
    """
    if graph_builder is None or state_cls is None:
        raise RuntimeError("visualize_and_explain: graph_builder와 state_cls를 주입하세요.")

    app = graph_builder(model)
    st = state_cls(
        user_question=question,
        json_data=json.dumps(data_rows, ensure_ascii=False),
        instruction=instruction or "",
    )

    # 그래프 실행 (네가 만든 langgraph: visualize → explain)
    out = app.invoke(st)

    # BaseModel 또는 Dict 모두 처리
    if hasattr(out, "json_graph"):
        json_graph = out.json_graph
        explanation = getattr(out, "explanation", "")
    elif isinstance(out, dict):
        json_graph = out.get("json_graph")
        explanation = out.get("explanation", "")
    else:
        json_graph = None
        explanation = ""

    if not json_graph:
        raise RuntimeError("visualize_and_explain: json_graph가 비었습니다.")

    return {"json_graph": json_graph, "explanation": explanation}
