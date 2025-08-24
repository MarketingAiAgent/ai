# orchestrator/tools/sql_executor.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.agents.text_to_sql.__init__ import call_sql_generator

logger = logging.getLogger(__name__)


def _ensure_table_payload(table: Any) -> Dict[str, Any]:
    """Normalize arbitrary table-like into {rows, columns, row_count}."""
    if not isinstance(table, dict):
        return {"rows": [], "columns": [], "row_count": 0}
    rows = table.get("rows") or []
    cols = table.get("columns") or []
    if isinstance(rows, list) and rows and isinstance(rows[0], dict) and not cols:
        cols = list(rows[0].keys())
    return {
        "rows": rows if isinstance(rows, list) else [],
        "columns": cols if isinstance(cols, list) else [],
        "row_count": int(table.get("row_count") or len(rows) or 0),
    }


def run_t2s_agent_with_instruction(state: Dict[str, Any], instruction: str) -> Dict[str, Any]:
    """Text-to-SQL 실행 (플래너가 만든 instruction 그대로 사용)."""
    result = call_sql_generator(
        message=instruction,
        conn_str=state["conn_str"],
        schema_info=state["schema_info"],
    )
    table = result.get("data_json")
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except Exception:
            logger.warning("data_json is not valid JSON; using empty table")
            table = {"rows": [], "columns": [], "row_count": 0}
    return _ensure_table_payload(table)


def execute_sql_from_plan(sql_plan: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    플래너 계획을 그대로 실행.
    sql_plan:
      { "enabled": true, "top_k": 3, "instruction": "..."}  # or "queries": [...]
    반환: table rows (상위 top_k까지만, 정렬/휴리스틱 없음)
    """
    if not sql_plan or not sql_plan.get("enabled"):
        return []
    instruction = sql_plan.get("instruction")
    if not instruction:
        qs = sql_plan.get("queries") or []
        instruction = qs[0] if qs else ""
    if not instruction:
        logger.warning("execute_sql_from_plan: empty instruction")
        return []
    table = run_t2s_agent_with_instruction(state, instruction)
    top_k = int(sql_plan.get("top_k") or 3)
    return (table.get("rows") or [])[: max(1, top_k)]
