from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import json 
import re 
import math

_TABLE_BLOCK_RE = re.compile(r"\[TABLE_START\].*?\[TABLE_END\]", re.DOTALL)

def _clean_text(text: str) -> str:
    if not text:
        return ""
    # 표 구간 제거 (프론트 약속 유지용 토큰은 그대로 두되, 본문 요약에선 제거)
    text = _TABLE_BLOCK_RE.sub("", text)
    # 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    # 문장 경계 우선: 마침표/물음표/감탄 부호 근처에서 자르기
    cut = s[:max_len]
    m = re.search(r"[\.?!]\s*\S*$", cut)
    if m:
        return cut[:m.start()+1].strip()
    return cut.rstrip() + "…"

def summarize_history(history: List[Dict[str, Any]], max_chars: int = 600) -> str:
    """
    최근 맥락을 규칙적으로 요약해 플래너에 전달합니다.
    - 최근 사용자 요청 1개 + 직전 요청 1개
    - 최근 어시스턴트 질문 1문장
    - 최근 제안 선택지(번호/불릿) 최대 5개
    - [TABLE_START]~[TABLE_END] 본문 제거
    """
    if not history:
        return ""

    # 1) 최신 순으로 스캔
    user_msgs: List[str] = []
    last_assistant_question: str = ""
    last_options: List[str] = []

    # 가장 최근 "어시스턴트" 메시지 하나를 먼저 잡아 옵션/질문 탐지에 사용
    last_assistant_msg = None
    for m in reversed(history):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant_msg = _clean_text(str(m["content"]))
            break

    if last_assistant_msg:
        # 질문 문장(물음표 또는 한국어 의문 종결) 한 줄 추출
        lines = [ln.strip() for ln in last_assistant_msg.splitlines() if ln.strip()]
        # 의문 조건: '?' 포함 또는 '까요/나요/죠/죠?' 등의 패턴
        q_pat = re.compile(r"(?:\?|겠[습니까]|까[요]?|나요|죠\?|죠$|어떤지|선택해 주|정해 주|골라 주)")
        for ln in lines:
            if q_pat.search(ln):
                last_assistant_question = ln
                break

        # 번호/불릿 선택지 추출 (1. / - / • / * 로 시작)
        opt_pat = re.compile(r"^(?:\d+\.\s+|[-•*]\s+)(.+)$")
        for ln in lines:
            m = opt_pat.match(ln)
            if m:
                cand = m.group(1).strip()
                # 너무 긴 라인은 자르기
                last_options.append(_truncate(cand, 80))
                if len(last_options) >= 5:
                    break

    # 2) 사용자 최근/직전 메시지
    for m in reversed(history):
        if m.get("role") == "user" and m.get("content"):
            user_msgs.append(_clean_text(str(m["content"])))
            if len(user_msgs) >= 2:
                break

    last_user = user_msgs[0] if len(user_msgs) >= 1 else ""
    prev_user = user_msgs[1] if len(user_msgs) >= 2 else ""

    # 3) 조립
    parts: List[str] = []
    if last_user:
        parts.append(f"최근 사용자 요청: { _truncate(last_user, 160) }")
    if prev_user:
        parts.append(f"이전 사용자 요청: { _truncate(prev_user, 120) }")
    if last_assistant_question:
        parts.append(f"최근에 제가 드린 질문: { _truncate(last_assistant_question, 120) }")
    if last_options:
        parts.append("제안했던 선택지: " + "; ".join(_truncate(o, 60) for o in last_options))

    summary = "\n".join(parts).strip()

    # 4) 길이 제한 보정
    if len(summary) <= max_chars:
        return summary

    # 너무 길면 각 파트별로 조금씩 더 잘라서 맞추기
    # (우선순위: 최근 사용자 > 최근 질문 > 이전 사용자 > 선택지)
    order = [0, 2, 1, 3]
    cuts = [160, 120, 120, 240]  # 초기 컷 기준
    while len(summary) > max_chars:
        changed = False
        for idx in order:
            if idx < len(parts) and len(parts[idx]) > 40:
                # 숫자만 조금씩 줄이기
                cuts[idx if idx < 3 else 3] = max(30, int(cuts[idx if idx < 3 else 3] * 0.9))
                # 다시 조립
                # 파트 재계산을 위해 원문을 다시 만들어야 하지만, 간단화: 전체를 더 강하게 truncate
                summary = _truncate(summary, max_chars)
                changed = True
                if len(summary) <= max_chars:
                    break
        if not changed:
            break

    return summary

def today_kr() -> str:
    """Asia/Seoul 기준 오늘 날짜 yyyy-mm-dd"""
    try:
        return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def normalize_table(table: Any) -> Dict[str, Any]:
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

def pick_col(columns: List[str], candidates: List[str]) -> Optional[str]:
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

def format_number(n: Any) -> str:
    try:
        x = float(n)
        # 정수처럼 보이면 정수로, 아니면 소수 2자리
        if abs(x - int(x)) < 1e-9:
            return f"{int(x):,}"
        return f"{x:,.2f}"
    except Exception:
        return str(n)

_NUMBER_STRIP_RE = re.compile(r"[,\s₩$€£]|(?<=\d)\%")

def to_float_safe(v: Any) -> Optional[float]:
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

def markdown_table(rows: List[Dict[str, Any]], columns: List[str], limit: int = 10) -> str:
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

def format_period_by_datecol(rows: List[Dict[str, Any]], date_col: Optional[str]) -> str:
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

def ensure_table_payload(payload: Any) -> Dict[str, Any]:
    """
    표준 계약(Contract)을 가볍게 검증/보정합니다.
    기대 포맷:
      { "rows": List[Dict], "columns": List[str], "row_count": int }
    - 불일치 시 빈 테이블로 반환
    - rows 길이와 row_count 불일치면 그대로 두고(호출자에서 의미), columns는 rows 첫 행 키로 보정
    """
    try:
        if not isinstance(payload, dict):
            return {"rows": [], "columns": [], "row_count": 0}
        rows = payload.get("rows", [])
        cols = payload.get("columns", [])
        rc = payload.get("row_count", 0)

        if not isinstance(rows, list):
            return {"rows": [], "columns": [], "row_count": 0}
        if rows and isinstance(rows[0], dict) and (not cols):
            cols = list(rows[0].keys())
        if not isinstance(cols, list):
            cols = []
        if not isinstance(rc, int):
            try:
                rc = int(rc)
            except Exception:
                rc = len(rows)
        return {"rows": rows, "columns": cols, "row_count": rc}
    except Exception:
        return {"rows": [], "columns": [], "row_count": 0}

GOOD_DIR = {  # 값이 클수록 좋은 지표
    "revenue": +1,
    "growth_pct": +1,
    "gm": +1,
    "conversion_rate": +1,
    "repeat_rate": +1,
    "aov": +1,
    "inventory_days": +1,
    "seasonality_score": +1,
    # 값이 작을수록 좋은 지표
    "return_rate": -1,
    "volatility_score": -1,
}

WEIGHTS = {
    "growth_pct": 0.35,
    "gm": 0.20,
    "conversion_rate": 0.15,
    "repeat_rate": 0.10,
    "seasonality_score": 0.10,
    "return_rate": 0.05,
    "inventory_days": 0.05,
    # 나머지는 있으면 보너스
}

def _collect_metric_vectors(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Tuple[float, float]]:
    """각 지표별 (min, max) 수집. 숫자만 고려."""
    stat: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            try:
                v = float(v)
                if not math.isnan(v):
                    vals.append(v)
            except Exception:
                continue
        if vals:
            stat[k] = (min(vals), max(vals))
    return stat

def _norm(value: Optional[float], minmax: Optional[Tuple[float, float]], dir_sign: int) -> Optional[float]:
    if value is None or minmax is None:
        return None
    lo, hi = minmax
    if hi - lo == 0:
        return 0.5
    x = (float(value) - lo) / (hi - lo)
    return x if dir_sign >= 0 else (1.0 - x)

def compute_opportunity_score(rows: List[Dict[str, Any]], trending_terms: List[str]) -> List[Dict[str, Any]]:
    """
    - rows: t2s가 반환한 레코드들(브랜드/카테고리/상품 레벨)
    - trending_terms: 외부 트렌딩 키워드 목록(라벨/이름 매칭 시 보너스)
    반환: 각 행에 scores/opportunity_score/reasons를 부착한 새 리스트
    """
    metrics = list(WEIGHTS.keys()) + ["revenue", "aov", "volatility_score"]
    stat = _collect_metric_vectors(rows, metrics)

    enriched = []
    for r in rows:
        scores: Dict[str, float] = {}
        reasons: List[str] = []

        for m, w in WEIGHTS.items():
            v = r.get(m)
            try:
                v = float(v)
            except Exception:
                v = None
            s = _norm(v, stat.get(m), GOOD_DIR.get(m, +1))
            if s is not None:
                scores[m] = s * w

        # 트렌딩 보너스(간단): 라벨 필드에서 용어 매칭 시 +0.05
        label = str(r.get("brand_name") or r.get("product_name") or r.get("category_name") or "")
        bonus = 0.0
        for t in trending_terms or []:
            if t and t.lower() in label.lower():
                bonus = 0.05
                reasons.append(f"트렌드 '{t}'와 매칭")
                break

        opp = sum(scores.values()) + bonus

        # 기본 이유 몇 가지 자동 생성(있을 때만)
        if r.get("growth_pct") is not None:
            reasons.append(f"증가율 {r.get('growth_pct')}%")
        if r.get("gm") is not None:
            reasons.append(f"마진 {r.get('gm')}")
        if r.get("inventory_days") is not None:
            reasons.append(f"재고여력 {r.get('inventory_days')}일")
        if r.get("return_rate") is not None:
            reasons.append(f"반품률 {r.get('return_rate')}")

        enriched.append({**r, "scores": scores, "opportunity_score": round(opp, 4), "reasons": reasons})

    return enriched

def pick_diverse_top_k(rows: List[Dict[str, Any]], k: int = 4) -> List[Dict[str, Any]]:
    """
    간단한 다양성 제약: category_name, price_band, gender_age 중 하나라도 다르게 유지하려 시도.
    """
    def tags(r):
        return (
            str(r.get("category_name") or ""),
            str(r.get("price_band") or ""),
            str(r.get("gender_age") or ""),
        )
    used = set()
    picked: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x.get("opportunity_score", 0), reverse=True):
        t = tags(r)
        if t in used and len(picked) < max(2, k-1):
            # 같은 태그면 스킵 시도, 그래도 부족하면 허용
            continue
        used.add(t)
        picked.append(r)
        if len(picked) >= k:
            break
    # 후보가 모자라면 높은 점수 순으로 보충
    if len(picked) < k:
        for r in sorted(rows, key=lambda x: x.get("opportunity_score", 0), reverse=True):
            if r not in picked:
                picked.append(r)
            if len(picked) >= k:
                break
    return picked