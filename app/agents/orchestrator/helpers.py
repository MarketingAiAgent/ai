from typing import List, Dict, Any, Optional 
from datetime import datetime
from zoneinfo import ZoneInfo
import json 
import re 

def _summarize_history(history: List[Dict[str, str]], limit_chars: int = 800) -> str:
    """최근 히스토리를 간단 요약으로 제공 (LLM 컨텍스트용)"""
    text = " ".join(h.get("content", "") for h in history[-6:])
    return text[:limit_chars]

def _today_kr() -> str:
    """Asia/Seoul 기준 오늘 날짜 yyyy-mm-dd"""
    try:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _normalize_table(table: Any) -> Dict[str, Any]:
    """
    t2s 표 결과를 표준 형태로 정규화.

    지원하는 입력:
      - {"rows":[...], "columns":[...]}                           # 이미 표준
      - {"columns":[...], "data":[[...], ...]}                    # pandas orient='split'
      - {"schema": {...}, "data":[{...}, ...]}                    # pandas orient='table'
      - [{"col": val, ...}, ...]                                  # pandas orient='records'
      - {col: {row_idx: val, ...}, ...}                           # pandas orient='columns'
      - {row_idx: {col: val, ...}, ...}                           # pandas orient='index'
      - [[...], [...]]                                            # 열 이름 미상 (col_0.. 생성)
    """
    # 문자열이면 JSON 먼저 파싱
    if isinstance(table, str):
        try:
            table = json.loads(table)
        except Exception:
            return {"rows": [], "columns": [], "row_count": 0}

    # 0) 이미 표준
    if isinstance(table, dict) and "rows" in table:
        rows = table.get("rows") or []
        cols = table.get("columns") or (list(rows[0].keys()) if rows and isinstance(rows[0], dict) else [])
        return {"rows": rows, "columns": cols, "row_count": len(rows)}

    # 1) split
    if isinstance(table, dict) and "columns" in table and "data" in table and isinstance(table["data"], list):
        cols = table["columns"]
        data = table["data"]
        rows = [{cols[i]: (row[i] if i < len(row) else None) for i in range(len(cols))} for row in data]
        return {"rows": rows, "columns": cols, "row_count": len(rows)}

    # 2) table
    if isinstance(table, dict) and "schema" in table and "data" in table and isinstance(table["data"], list):
        data = table["data"]
        if data and isinstance(data[0], dict):
            cols = list(data[0].keys())
            return {"rows": data, "columns": cols, "row_count": len(data)}

    # 3) records
    if isinstance(table, list) and (not table or isinstance(table[0], dict)):
        cols = list(table[0].keys()) if table else []
        return {"rows": table, "columns": cols, "row_count": len(table)}

    # 4) columns (dict-of-dicts or dict-of-lists)
    if isinstance(table, dict) and table and all(isinstance(v, (dict, list)) for v in table.values()):
        cols = list(table.keys())
        # dict-of-dicts: {col: {row_idx: val}}
        if all(isinstance(v, dict) for v in table.values()):
            row_keys = set()
            for d in table.values():
                row_keys |= set(d.keys())

            def _ord(k):
                try: return int(k)
                except Exception:
                    try: return float(k)
                    except Exception:
                        return str(k)

            ordered = sorted(list(row_keys), key=_ord)
            rows = []
            for rk in ordered:
                row = {c: table[c].get(rk) for c in cols}
                rows.append(row)
            return {"rows": rows, "columns": cols, "row_count": len(rows)}

        # dict-of-lists: {col: [v0, v1, ...]}
        if all(isinstance(v, list) for v in table.values()):
            maxlen = max((len(v) for v in table.values()), default=0)
            rows = [{c: (table[c][i] if i < len(table[c]) else None) for c in cols} for i in range(maxlen)]
            return {"rows": rows, "columns": cols, "row_count": len(rows)}

    # 5) index orientation: {row_idx: {col: val}}
    if isinstance(table, dict) and table and all(isinstance(v, dict) for v in table.values()):
        try:
            items = list(table.items())

            def _ord2(k):
                try: return int(k)
                except Exception:
                    try: return float(k)
                    except Exception:
                        return str(k)

            items.sort(key=lambda kv: _ord2(kv[0]))
            rows = [kv[1] for kv in items]
            cols = list(rows[0].keys()) if rows else []
            return {"rows": rows, "columns": cols, "row_count": len(rows)}
        except Exception:
            pass

    # 6) list-of-lists
    if isinstance(table, list) and table and isinstance(table[0], (list, tuple)):
        max_cols = max((len(r) for r in table), default=0)
        cols = [f"col_{i}" for i in range(max_cols)]
        rows = [{cols[i]: v for i, v in enumerate(r)} for r in table]
        return {"rows": rows, "columns": cols, "row_count": len(rows)}

    return {"rows": [], "columns": [], "row_count": 0}

def _pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lc = [c.lower() for c in columns]
    for cand in candidates:
        if cand.lower() in lc:
            return columns[lc.index(cand.lower())]
    # 부분 일치도 허용(예: 'product_name', 'product')
    for i, c in enumerate(lc):
        for cand in candidates:
            if cand.lower() in c:
                return columns[i]
    return None

def _format_number(n: Any) -> str:
    try:
        x = float(n)
        # 정수처럼 보이면 정수로, 아니면 소수 2자리
        if abs(x - int(x)) < 1e-9:
            return f"{int(x):,}"
        return f"{x:,.2f}"
    except Exception:
        return str(n)

_NUMBER_STRIP_RE = re.compile(r"[,\s₩$€£]|(?<=\d)\%")

def _to_float_safe(v: Any) -> Optional[float]:
    """문자·통화·퍼센트 등을 안전하게 float 변환"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    # 괄호 음수 (예: (1,234))
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = _NUMBER_STRIP_RE.sub("", s)
    try:
        x = float(s)
        return -x if neg else x
    except Exception:
        return None

def _markdown_table(rows: List[Dict[str, Any]], columns: List[str], limit: int = 10) -> str:
    if not rows:
        return "_표시할 데이터가 없습니다._"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows[:limit]:
        line = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        lines.append(line)
    if len(rows) > limit:
        lines.append(f"\n_표시는 상위 {limit}행 미리보기입니다 (총 {len(rows)}행)._")
    return "\n".join(lines)

def _format_period_by_datecol(rows: List[Dict[str, Any]], date_col: Optional[str]) -> str:
    """date 열이 있으면 min~max 기간을 표시"""
    if not rows or not date_col:
        return ""
    vals = []
    for r in rows:
        v = r.get(date_col)
        if v is None:
            continue
        s = str(v)
        # 단순 파싱 (YYYY-MM-DD, YYYY/MM/DD, YYYYMMDD 등)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m"):
            try:
                d = datetime.strptime(s[:len(fmt)], fmt)
                vals.append(d)
                break
            except Exception:
                continue
    if not vals:
        return ""
    start = min(vals).strftime("%Y-%m-%d")
    end = max(vals).strftime("%Y-%m-%d")
    return f" (기간: {start} ~ {end})"
