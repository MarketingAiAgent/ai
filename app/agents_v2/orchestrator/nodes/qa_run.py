from __future__ import annotations

import logging
from typing import Any, Dict

from app.agents_v2.orchestrator.state import AgentState
from orchestrator.tools import (
    run_t2s_agent_with_instruction,
    execute_web_from_plan,
    visualize_and_explain,
)

logger = logging.getLogger(__name__)


def qa_run_node(
    state: AgentState,
    *,
    sql_context: Dict[str, Any],
    viz_builder=None,
    viz_state_cls=None,
) -> AgentState:
    """
    입력:
      state = { "user_message": str, "qa_plan": {...} }
      sql_context = {"conn_str": ..., "schema_info": ...}
      viz_builder = build_visualize_graph (선택)
      viz_state_cls = VisualizeState (선택)
    출력: state에 다음 키 추가
      - "qa_table": Dict(rows/columns/row_count)        # T2S 결과(있을 때)
      - "qa_chart": Optional[str]                        # Plotly JSON
      - "qa_explanation": Optional[str]                  # 시각화 해설
      - "qa_web_rows": Optional[List[dict]]              # web 실행 결과(name/signal/source)
      - "qa_snapshot": Optional[Dict]                    # 답변 노드용 간단 요약
    """
    plan = state.qa_plan or {}
    choice = (plan.choice or "t2s").lower()
    t2s = plan.t2s or {}
    web = plan.web or {}

    if choice in ("t2s", "both") and t2s.enabled:
        inst = t2s.instruction or ""
        table = run_t2s_agent_with_instruction(sql_context, inst)
        state["qa_table"] = table

        if t2s.visualize and viz_builder and viz_state_cls:
            try:
                viz = visualize_and_explain(
                    question=state.user_message or "",
                    data_rows=table.get("rows") or [],
                    instruction=t2s.viz_hint or "",
                    model="gemini-2.5-flash",
                    graph_builder=viz_builder,
                    state_cls=viz_state_cls,
                )
                state["qa_chart"] = viz.json_graph
                state["qa_explanation"] = viz.explanation or ""
            except Exception as e:
                logger.exception("[qa_run_node] visualize_and_explain 실패: %s", e)

    # ---- Web (외부 지식) ----
    if choice in ("web", "both") and web.enabled:
        try:
            web_rows = execute_web_from_plan({
                "enabled": True,
                "query": web.query or (web.queries or [None])[0],
                "queries": [],
                "use_sources": web.use_sources or ["supabase_marketing","supabase_beauty","tavily"],
                "top_k": int(web.top_k or 5),
                "scrape_k": int(web.scrape_k or 0),
            })
            state["qa_web_rows"] = web_rows

            # 답변 노드 호환용 간단 스냅샷
            notes   = [r["name"] for r in web_rows if r.name]
            sources = [{"title": r.name, "url": r.source} for r in web_rows if r.name]
            state["qa_snapshot"] = {"trending_terms": [], "notes": notes[:5], "sources": sources[:5]}
        except Exception as e:
            logger.exception("[qa_run_node] web executor 실패: %s", e)
            state["qa_snapshot"] = {"trending_terms": [], "notes": [str(e)], "sources": []}

    return state
