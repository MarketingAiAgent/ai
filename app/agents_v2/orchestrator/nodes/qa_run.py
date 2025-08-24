from __future__ import annotations

import logging
from typing import Any, Dict

from app.agents_v2.orchestrator.state import AgentState
from app.agents_v2.orchestrator.tools import (
    run_t2s_agent_with_instruction,
    execute_web_from_plan,
    visualize_and_explain,
)
from app.agents_v2.visualizer.graph import build_visualize_graph
from app.agents_v2.visualizer.state import VisualizeState

logger = logging.getLogger(__name__)


def qa_run_node(state: AgentState) -> AgentState:
    """
    입력 state:
      - qa_plan: QAPlan
      - sql_context: Dict[str, Any] (conn_str, schema_info)
    출력: state에 다음 키 추가
      - qa_table: Dict(rows/columns/row_count)        # T2S 결과(있을 때)
      - qa_chart: Optional[str]                        # Plotly JSON
      - qa_explanation: Optional[str]                  # 시각화 해설
      - qa_web_rows: Optional[List[dict]]              # web 실행 결과(name/signal/source)
      - qa_snapshot: Optional[Dict]                    # 답변 노드용 간단 요약
    """
    plan = state.qa_plan
    sql_context = state.sql_context
    if not plan or not sql_context:
        return state
    
    choice = (plan.choice or "t2s").lower()
    t2s = plan.t2s
    web = plan.web

    updates = {}
    
    if choice in ("t2s", "both") and t2s and t2s.enabled:
        inst = t2s.instruction or ""
        table = run_t2s_agent_with_instruction(sql_context, inst)
        updates["qa_table"] = table

        if t2s.visualize:
            try:
                viz = visualize_and_explain(
                    question=state.user_message or "",
                    data_rows=table.get("rows", []) if table else [],
                    instruction=t2s.viz_hint or "",
                    model="gemini-2.5-flash",
                    graph_builder=build_visualize_graph,
                    state_cls=VisualizeState,
                )
                updates["qa_chart"] = viz["json_graph"]
                updates["qa_explanation"] = viz.get("explanation", "")
            except Exception as e:
                logger.exception("[qa_run_node] visualize_and_explain 실패: %s", e)

    # ---- Web (외부 지식) ----
    if choice in ("web", "both") and web and web.enabled:
        try:
            web_rows = execute_web_from_plan(web)
            updates["qa_web_rows"] = web_rows

            # 답변 노드 호환용 간단 스냅샷
            notes   = [r["name"] for r in web_rows if r.get("name")]
            sources = [{"title": r.get("name", ""), "url": r.get("source", "")} for r in web_rows if r.get("name")]
            updates["qa_snapshot"] = {"trending_terms": [], "notes": notes[:5], "sources": sources[:5]}
        except Exception as e:
            logger.exception("[qa_run_node] web executor 실패: %s", e)
            updates["qa_snapshot"] = {"trending_terms": [], "notes": [str(e)], "sources": []}

    return state.model_copy(update=updates)
