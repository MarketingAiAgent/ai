# agent/nodes/generate_promotion_report_node.py
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.agents_v2.orchestrator.state import PromotionSlots

logger = logging.getLogger(__name__)
ScopeLiteral = Literal["브랜드", "제품"]


# =========================
# Pydantic outputs
# =========================

class PromotionReport(BaseModel):
    title: str
    summary: str
    slots_recap: Dict[str, Optional[str]]  # audience/scope/target/period/KPI/concept
    highlights: List[str] = Field(default_factory=list)  # SQL/WEB 근거 요약 bullets
    plan: Dict[str, Any] = Field(
        default_factory=dict,
        description="실행 계획(예: concept, key_channels, offers, audience_notes 등)"
    )
    kpis: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    markdown: str = Field(description="프론트에 바로 렌더 가능한 마크다운 본문")


class ReportNodeOutput(BaseModel):
    message: str  # 한 줄 알림/확인 질문
    report: PromotionReport
    expect_fields: List[str] = Field(default_factory=list)


# =========================
# Evidence collection helpers
# =========================

_NAME_KEYS = ("name", "label", "target", "brand", "브랜드", "product", "제품", "타겟")

def _coerce_name(row: Dict[str, Any]) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for k in _NAME_KEYS:
        if k in row and row[k]:
            return str(row[k]).strip()
    # fallback: 첫 non-empty 값
    for _, v in row.items():
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

def _sql_bullet(row: Dict[str, Any]) -> Optional[str]:
    # rationale 우선
    rat = row.get("rationale")
    if isinstance(rat, str) and rat.strip():
        return rat.strip()[:180]
    # 간략 숫자형 1~2개 요약
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
            numeric_items.append(f"{str(k)[:16]}={val}")
        except Exception:
            continue
    if numeric_items:
        return "핵심지표: " + ", ".join(numeric_items[:2])
    return None

def _collect_insight_bullets(sql_rows: List[Dict[str, Any]], web_rows: List[Dict[str, Any]], top_k: int = 6) -> List[str]:
    bullets: List[str] = []
    # SQL 근거
    for r in sql_rows or []:
        nm = _coerce_name(r)
        b = _sql_bullet(r)
        if nm and b:
            bullets.append(f"{nm}: {b}")
        elif b:
            bullets.append(b)
    # WEB 근거
    for w in web_rows or []:
        name = (w.get("name") or "").strip()
        sig = (w.get("signal") or "").strip()
        if name and sig:
            bullets.append(f"{name}: {sig}")
        elif sig:
            bullets.append(sig)
    # 중복/정리
    seen = set()
    uniq: List[str] = []
    for b in bullets:
        k = b[:100]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(b)
        if len(uniq) >= top_k:
            break
    return uniq


# =========================
# Prompt builder
# =========================

def _build_report_messages(
    slots: PromotionSlots,
    scope: ScopeLiteral,
    insight_bullets: List[str],
) -> List[tuple]:
    """
    LLM에게 '구조화 리포트 + 마크다운'을 JSON으로 생성시키는 프롬프트.
    - 컬럼/스키마 지시 금지
    - 입력 슬롯을 요약/정렬해 리포트에 반영
    - 근거 bullets 반영
    """
    system = (
        "당신은 한국어 마케팅 기획 리포트 작성 어시스턴트입니다. "
        "입력된 Promotion Slots와 참고 근거(bullets)를 바탕으로 간결하지만 실행 가능한 리포트를 만듭니다. "
        "반드시 아래 JSON 스키마로만 출력하세요. 마크다운 본문(markdown)은 헤더/리스트를 활용해 읽기 좋게 작성하세요.\n\n"
        "출력 스키마:\n"
        "{\n"
        '  "message": "한 줄 알림 또는 최종 확인 질문",\n'
        '  "report": {\n'
        '    "title": "제목",\n'
        '    "summary": "요약",\n'
        '    "slots_recap": {"audience": "...","scope": "...","target": "...","period": "...","KPI": "...","concept": "..."},\n'
        '    "highlights": ["근거 bullet", "..."],\n'
        '    "plan": {"concept": "...", "key_channels": ["..."], "offers": ["..."], "audience_notes": "...", "timeline": "..."},\n'
        '    "kpis": ["제안 KPI 2~4개(입력 KPI가 있으면 우선 반영)"],\n'
        '    "risks": ["리스크 2~3개와 간단 대응"],\n'
        '    "next_steps": ["다음 단계 2~4개"],\n'
        '    "markdown": "최종 리포트 본문 마크다운"\n'
        "  },\n"
        '  "expect_fields": []\n'
        "}\n"
        "규칙:\n"
        "1) tone은 간결/실무 중심. 숫자·고유명사는 과장하지 말고 입력 bullets 범위에서만 활용.\n"
        "2) scope가 '브랜드'면 타겟을 브랜드 관점으로, '제품'이면 제품 관점으로 서술.\n"
        "3) period를 '일정/운영' 섹션에 자연스럽게 반영.\n"
        "4) KPI가 입력에 없으면 합리적인 KPI(예: 전환률/신규구매수/리치 등) 2~4개 제안.\n"
        "5) markdown에는 섹션 헤더(H2/H3), 목록, 굵게 등을 활용. 표는 필요할 때만.\n"
        "6) JSON 외 텍스트는 출력하지 마세요."
    )

    user_payload = {
        "slots": slots.model_dump(),
        "scope": scope,
        "insights": insight_bullets,
    }
    return [("system", system), ("human", json.dumps(user_payload, ensure_ascii=False))]


# =========================
# Node
# =========================

def generate_promotion_report_node(state: Dict) -> Dict:
    """
    입력 state:
      {
        "slots": PromotionSlots(dict),        # 필수
        "sql_rows": List[dict] (선택),
        "web_rows": List[dict] (선택)
      }
    출력:
      {
        "message": str,                       # "리포트를 정리했습니다…" 등
        "report": PromotionReport(dict),      # 구조화 객체
        "report_markdown": str,               # 프론트에서 바로 렌더
        "expect_fields": []
      }
    """
    logger.info("===== 🚀 프로모션 리포트 생성 노드 실행 =====")
    slots_dict: Dict = state.get("slots") or {}
    sql_rows: List[Dict[str, Any]] = state.get("sql_rows") or []
    web_rows: List[Dict[str, Any]] = state.get("web_rows") or []

    # 슬롯 검증
    try:
        slots = PromotionSlots.model_validate(slots_dict)
    except ValidationError:
        slots = PromotionSlots()

    # 필수 충족 여부: scope/period/target이 없으면 리포트 불가
    missing: List[str] = []
    if not slots.scope:
        missing.append("scope(브랜드/제품)")
    if not slots.period:
        missing.append("period(기간)")
    if not slots.target:
        missing.append("target(대상 브랜드/제품)")

    if missing:
        return {
            "message": f"리포트를 생성하려면 다음 정보가 필요합니다: {', '.join(missing)}",
            "report": {},
            "report_markdown": "",
            "expect_fields": [],  # 상위 그래프에서 ASK 노드로 연결
        }

    # 근거 bullets 수집
    insight_bullets = _collect_insight_bullets(sql_rows, web_rows, top_k=6)

    # LLM 호출
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY)
    messages = _build_report_messages(slots, slots.scope, insight_bullets)
    prompt = ChatPromptTemplate.from_messages(messages)
    parser = PydanticOutputParser(pydantic_object=ReportNodeOutput)

    try:
        out: ReportNodeOutput = (prompt | llm | parser).invoke({})

        md = out.report.markdown.strip() if out.report and out.report.markdown else ""
        if not md:
            recap = out.report.slots_recap if out.report else {}
            high = out.report.highlights if out.report else []
            md_lines = [
                f"# {out.report.title if out.report else '프로모션 리포트'}",
                "",
                "## 개요",
                out.report.summary if out.report else "",
                "",
                "## 슬롯 요약",
                f"- 대상: {recap.get('audience')}",
                f"- 스코프: {recap.get('scope')}",
                f"- 타겟: {recap.get('target')}",
                f"- 기간: {recap.get('period')}",
                f"- KPI: {recap.get('KPI')}",
                f"- 컨셉: {recap.get('concept')}",
                "",
                "## 근거 하이라이트",
                *[f"- {b}" for b in high],
            ]
            md = "\n".join(md_lines)

        return {
            "message": out.message or "프로모션 리포트를 정리했습니다. 검토 후 확정해 주세요.",
            "report": out.report.model_dump(),
            "report_markdown": md,
            "expect_fields": [],
        }

    except Exception:
        logger.exception("[generate_promotion_report_node] LLM 실패 → 폴백 리포트 생성")
        # 간단 폴백 리포트
        bullets = insight_bullets or ["(참고 근거가 충분하지 않습니다. 내부 데이터/트렌드 연동을 권장합니다.)"]
        title = f"{slots.period} | {slots.scope} '{slots.target}' 프로모션 제안"
        md = "\n".join([
            f"# {title}",
            "",
            "## 개요",
            f"- 대상: {slots.audience or '미지정'}",
            f"- 스코프: {slots.scope}",
            f"- 타겟: {slots.target}",
            f"- 기간: {slots.period}",
            f"- KPI: {slots.KPI or '미지정'}",
            f"- 컨셉: {slots.concept or '미지정'}",
            "",
            "## 근거 하이라이트",
            *[f"- {b}" for b in bullets],
            "",
            "## 실행 계획 (초안)",
            "- 채널: 유료/소셜/크리에이터 중 타겟 특성에 맞춰 조합",
            "- 오퍼: 신규 첫구매/번들/한정기간 혜택 등",
            "- 크리에이티브: 타겟 언어/밈/숏폼 활용",
            "",
            "## 리스크 & 대응",
            "- 성과 변동성: 캘린더 상 피크/오프피크 고려",
            "- 재고/공급: 오퍼 과다 시 품절 위험 → 단계적 보정",
            "",
            "## 다음 단계",
            "- 예산/채널 배분 확정",
            "- 크리에이티브 콘셉트 샘플 공유",
            "- 추적 KPI/계측 이벤트 점검",
        ])
        return {
            "message": "프로모션 리포트 초안을 생성했습니다. 세부 조정이 필요하면 말씀해 주세요.",
            "report": PromotionReport(
                title=title,
                summary="핵심 슬롯과 참고 근거를 바탕으로 구성한 초안입니다.",
                slots_recap={
                    "audience": slots.audience,
                    "scope": slots.scope,
                    "target": slots.target,
                    "period": slots.period,
                    "KPI": slots.KPI,
                    "concept": slots.concept,
                },
                highlights=bullets,
                plan={"concept": slots.concept or "", "timeline": slots.period},
                kpis=[slots.KPI] if slots.KPI else ["전환수/매출", "CTR/도달", "신규구매수"],
                risks=["성과 변동성", "재고/공급 리스크"],
                next_steps=["예산/채널 배분 확정", "크리에이티브 샘플 작성", "추적 KPI 설정"],
                markdown=md,
            ).model_dump(),
            "report_markdown": md,
            "expect_fields": [],
        }
