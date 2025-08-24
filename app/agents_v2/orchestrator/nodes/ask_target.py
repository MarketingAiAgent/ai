# agent/nodes/build_options_and_question_node.py
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Literal, Tuple

from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.agents_v2.orchestrator.state    import PromotionSlots
from app.agents_v2.orchestrator.state    import AgentState

logger = logging.getLogger(__name__)
ScopeLiteral = Literal["브랜드", "제품"]


# ===== 출력 모델 =====
class OptionCandidate(BaseModel):
    label: str
    reason: str
    concept_suggestion: Optional[str] = None
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])

class AskTargetOutput(BaseModel):
    """
    LLM이 '질문 문장'과 '옵션 리스트'를 제안.
    - message: 헤더성 질문(한 문장)
    - options: 옵션 n개 (label/reason/optional concept_suggestion)
    """
    message: str
    options: List[OptionCandidate]
    expect_fields: List[Literal["target"]] = Field(default_factory=lambda: ["target"])


# ===== 유틸: SQL/WEB 증거 병합 (스키마 모름 전제) =====
# name 후보 키(최소 휴리스틱)
_NAME_KEYS = ("name", "label", "target", "brand", "브랜드", "product", "제품", "타겟")

def _coerce_name(row: Dict[str, Any]) -> Optional[str]:
    """SQL 행에서 표시용 이름을 뽑는다. 우선 name 계열 키, 없으면 첫 non-empty 값."""
    if not isinstance(row, dict):
        return None
    for k in _NAME_KEYS:
        if k in row and row[k]:
            return str(row[k]).strip()
    # 첫 non-empty 값
    for _, v in row.items():
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

def _sql_reason_from_row(row: Dict[str, Any]) -> Optional[str]:
    """
    SQL 행에서 사람이 읽을 간단 설명을 만든다.
    - 우선 'rationale' 텍스트가 있으면 그걸 사용
    - 없으면 숫자형/비율형 칼럼 1~2개를 'k=v' 요약으로 생성
    """
    if not isinstance(row, dict):
        return None

    # 1) rationale 우선
    rat = row.get("rationale")
    if isinstance(rat, str) and rat.strip():
        return rat.strip()[:160]

    # 2) 숫자형 1~2개 요약
    numeric_items: List[str] = []
    for k, v in row.items():
        if k in _NAME_KEYS or k == "rationale" or v is None:
            continue
        try:
            fv = float(v)
            if abs(fv) >= 1_000_000:
                val = f"{fv:,.0f}"
            elif abs(fv) >= 1_000:
                val = f"{fv:,.0f}"
            else:
                val = f"{fv:.2f}".rstrip("0").rstrip(".")
            k_disp = str(k)[:16]
            numeric_items.append(f"{k_disp}={val}")
        except Exception:
            continue

    if numeric_items:
        return "핵심지표: " + ", ".join(numeric_items[:2])
    return None

def _merge_evidences(
    sql_rows: List[Dict[str, Any]],
    web_rows: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    동일 타겟명(name)을 기준으로 SQL/WEB 신호를 합친다.
    출력 예:
      { "name": "브랜드 A",
        "bullets": ["핵심지표: 매출증가율=0.24", "인기 유튜버 협업 120만 조회"],
        "sources": ["...","..."] }
    """
    by_name: Dict[str, Dict[str, Any]] = {}

    # SQL -> name + reason
    for row in (sql_rows or []):
        name = _coerce_name(row)
        if not name:
            continue
        node = by_name.setdefault(name, {"name": name, "bullets": [], "sources": []})
        sql_reason = _sql_reason_from_row(row)
        if sql_reason:
            node["bullets"].append(sql_reason)

    # WEB -> name + signal
    for r in (web_rows or []):
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        node = by_name.setdefault(name, {"name": name, "bullets": [], "sources": []})
        signal = str(r.get("signal") or "").strip()
        source = str(r.get("source") or "").strip()
        if signal:
            node["bullets"].append(signal[:180])
        if source:
            node["sources"].append(source)

    # 간단 정렬: bullet 수 ↓, name 가나다/알파
    items = list(by_name.values())
    items.sort(key=lambda x: (-len(x.get("bullets", [])), x["name"]))
    return items[: max(1, top_k + 2)]  # 여유 2개


# ===== 프롬프트 =====
def _build_question_messages(
    scope: ScopeLiteral,
    audience: Optional[str],
    evidences: List[Dict[str, Any]],
    top_k: int,
) -> List[tuple]:
    """
    LLM이 evidence를 받아 **질문 문구 + 옵션**을 생성한다.
    - options는 최대 top_k개
    - reason은 1줄 요약
    - concept_suggestion은 선택(없으면 생략 가능)
    """
    system = (
        "당신은 한국어로 선택지 질문을 작성하는 조교입니다. "
        "입력으로 scope(브랜드/제품), audience(있을 수도 없음), evidences(후보별 근거)를 받고 "
        "오직 아래 JSON만 반환하세요:\n"
        "{\n"
        '  "message": "한 문장 질문(존댓말)",\n'
        '  "options": [\n'
        '     {{"label":"타겟명","reason":"1줄 근거","concept_suggestion":"선택"}},\n'
        '     {{"label":"...", "reason":"...", "concept_suggestion":"선택"}}\n'
        "  ],\n"
        '  "expect_fields": ["target"]\n'
        "}\n"
        "규칙:\n"
        f"1) 옵션은 최대 {top_k}개로 제한하고, 가장 설득력 있는 후보를 우선으로 고르세요.\n"
        "2) reason은 evidences의 bullets를 바탕으로 1줄 한국어 요약으로 만드세요(불필요한 수치 나열 금지).\n"
        "3) audience가 있다면 reason에 자연스럽게 반영하세요(예: 20대 관련 맥락).\n"
        "4) JSON 외 텍스트 출력 금지."
    )

    # evidences → 간결 구조로 축약
    mini = [{"name": ev["name"], "bullets": ev.get("bullets", [])[:3]} for ev in evidences[: max(1, top_k * 2)]]

    user_payload = {
        "scope": scope,
        "audience": audience,
        "evidences": mini,
        "top_k": top_k,
    }
    return [("system", system), ("human", json.dumps(user_payload, ensure_ascii=False))]


# ===== 메시지 포맷터 =====
def _format_message_with_options(
    header_sentence: str,
    scope: ScopeLiteral,
    audience: Optional[str],
    options: List[OptionCandidate],
) -> str:
    """
    채팅에 바로 출력할 메시지 문자열을 구성한다.
    예시 포맷:
      "20대 대상, 브랜드 기준 추천입니다.
       1) 브랜드 A — 이유... · 컨셉: ...
       2) 브랜드 B — 이유...
       직접 입력도 가능합니다..."
    """
    header = header_sentence.strip()
    if not header:
        # 안전 헤더
        aud = f"{audience} 대상, " if audience else ""
        header = f"{aud}{scope} 기준 추천입니다."

    lines = [header, ""]
    for i, o in enumerate(options, start=1):
        base = f"{i}) {o.label} — {o.reason}"
        if o.concept_suggestion:
            base += f" · 컨셉: {o.concept_suggestion}"
        lines.append(base)
    lines.append("")
    lines.append("직접 입력도 가능합니다. 어느 타겟으로 진행하시겠습니까? (예: '1번', '브랜드 A')")
    return "\n".join(lines)


# ===== 노드 본체 =====
def build_options_and_question_node(state: AgentState) -> AgentState:
    """
    입력 state:
      - promotion_slots: PromotionSlots
      - sql_rows: List[dict]   # 실행기 결과(표의 각 행). 첫 컬럼은 name이길 권장
      - web_rows: List[dict]   # 실행기 결과(name/signal/source)
      - top_k: int (선택, 기본 3)
    출력:
      - response: str          # 질문 + 옵션을 포함한 완성 메시지
      - options: List[OptionCandidate]
      - expect_fields: List[str]
    """
    logger.info("===== 🚀 타겟 후보 추천 노드 실행 =====")
    
    slots = state.promotion_slots
    sql_rows = state.sql_rows or []
    web_rows = state.web_rows or []
    top_k = 3  # 기본값

    # 전제: scope/period가 있어야 옵션을 물을 타이밍
    if not slots or not slots.scope or not slots.period:
        return state.model_copy(update={
            "response": "스코프(브랜드/제품)와 기간을 먼저 알려주시면 타겟 후보를 추천드리겠습니다.",
            "expect_fields": ["scope", "period"],
        })

    # 증거 병합
    evidences = _merge_evidences(sql_rows, web_rows, top_k=top_k)
    if not evidences:
        return state.model_copy(update={
            "response": "현재 추천할 타겟 후보를 찾지 못했습니다. 직접 브랜드/제품명을 입력해 주시겠어요?",
            "expect_fields": ["target"],
        })

    # LLM 호출(질문/옵션 생성)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_messages(
        _build_question_messages(slots.scope, slots.audience, evidences, top_k)
    )
    parser = PydanticOutputParser(pydantic_object=AskTargetOutput)

    try:
        out: AskTargetOutput = (prompt | llm | parser).invoke({})
        # 상한 적용
        options = out.options[:top_k]

        # 메시지 최종 구성(헤더 + 번호 매긴 옵션 + 선택 예시)
        final_message = _format_message_with_options(out.message, slots.scope, slots.audience, options)

        return state.model_copy(update={
            "response": final_message,
            "options": options,
            "expect_fields": ["target"],
        })
    except Exception:
        logger.exception("[build_options_and_question_node] LLM 실패 → 폴백 메시지 사용")
        # 폴백: evidences로 간단 옵션 구성
        opts: List[OptionCandidate] = []
        for ev in evidences[:top_k]:
            bullets = ev.get("bullets", [])
            reason = (bullets[0] if bullets else "관련 지표/트렌드 근거")[:160]
            opts.append(OptionCandidate(label=ev["name"], reason=reason))

        header = f"{slots.scope} 기준 추천입니다."
        final_message = _format_message_with_options(header, slots.scope, slots.audience, opts)

        return state.model_copy(update={
            "response": final_message,
            "options": opts,
            "expect_fields": ["target"],
        })
