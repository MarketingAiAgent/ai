from langchain_google_genai import ChatGoogleGenerativeAI

import datetime 
import re
import json
from zoneinfo import ZoneInfo

from supabase import create_client, Client

from app.core.config import settings
from .state import *

# ===== Helper =====

_OPEN = re.compile(r"^```[a-zA-Z0-9_-]*\r?\n")
_CLOSE = re.compile(r"\n?```\s*$")
def _strip_code_fence(s:str)->str:
     s = _OPEN.sub("", s.strip())
     return _CLOSE.sub("", s).strip()
def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    s = str(x).strip()
    return [s] if s else []



# ===== Prompt =====
LLM_PARSE_PROMPT = f"""
역할: 자유 형식 마케팅 입력을 구조화한다. 라벨/순서/언어/불릿/누락 모두 가능.
출력은 오직 JSON 하나. 코드펜스/설명 금지.
오늘 날짜 컨텍스트: TODAY={datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()} (Asia/Seoul)

스키마:
{{
  "category": string[],
  "target": string[],
  "period_raw": string[],        // 입력에서 보인 기간 관련 표현(원문 조각들)
  "products": string[],
  "brand": string[],
  "insight": string,
  "normalized_period": string,   // 정책:
                                 // 1) 입력에 명시적 날짜 범위가 있으면 'MM.DD ~ MM.DD'로 표준화 (예: '08.12 ~ 08.18')
                                 // 2) 날짜 범위가 없지만 기간 라벨(월/분기/시즌/주 등)이 있으면, TODAY와 라벨 의미에 부합하는
                                 //    "현실적이고 근접한" 7일 연속 구간을 네가 스스로 선택해 'MM.DD ~ MM.DD'로 제시한다.
                                 //    - 예: '여름'이면 현재 날짜 기준 여름에 해당하는 합리적 7일
                                 //    - 예: '8월'이면 8월 안에서 가까운 7일
                                 //    - 예: '이번 주'이면 해당 주의 7일
                                 // 3) 기간 단서가 전혀 없으면 "" (비워 둔다)
  "title_suggestion": string,    // 입력 맥락 기반 간결 타이틀 제안. 애매하면 "".
  "theme_labels": string[],      // 입력에서 추출한 테마 키워드(사전 제한 없음)
  "confidence": {{
    "category": number, "target": number, "period_raw": number,
    "products": number, "brand": number, "insight": number,
    "normalized_period": number, "title_suggestion": number, "theme_labels": number
  }},
  "assumptions": string          // normalized_period 결정을 위한 가정/기준을 한 줄 요약
}}

지침:
- 명시적 날짜가 있으면 그대로 표준화해 normalized_period에 넣는다.
- 날짜가 없고 라벨만 있으면, TODAY와 라벨 의미를 고려해 네가 "결정"한 7일 범위를 'MM.DD ~ MM.DD'로 제시한다.
- 기간 단서가 전혀 없으면 normalized_period는 ""(빈 문자열)로 둔다.
- 값이 하나라도 리스트로 감싸고, 공백/중복/장식 문자는 정리한다.
- 지정 키 외 어떤 것도 출력하지 마라.

[입력]
{{TEXT}}
""".strip()

LLM_DECIDE_PLAN_PROMPT = """
역할: 아래 요약을 보고 더 적합한 플랜 타입을 선택하여 JSON으로만 답하라.
선택지는 단 두 가지: "단일 프로모션" 또는 "카테고리/계절 프로모션".
설명 금지. 코드펜스 금지.

스키마:
{"plan_type": "단일 프로모션" | "카테고리/계절 프로모션"}

판단 힌트:
- 특정 브랜드 단독/한정 SKU 중심 → 단일 프로모션
- 카테고리/계절/테마 묶음/브랜드 다수 → 카테고리/계절 프로모션

[요약]
category={category}
target={target}
products={products}
brand={brand}
theme_labels={theme_labels}
insight={insight}
""".strip()

def _brand_prompt(parsed: Dict[str, Any], example: List[Dict[str, Any]], ref: Dict[str, Any] | None = None) -> str:
    p = parsed
    primary_brand = (p.get("brand") or [""])[0]
    ref = ref or {}
    ref_title = ref.get("title","")
    ref_main = ref.get("main_banner","")
    ref_coupon = ref.get("coupon_section","")
    ref_product = ref.get("product_section","")
    ref_notes = ref.get("event_notes","")

    return f"""
너는 올리브영 '브랜드 프로모션' JSON을 작성하는 카피라이터다.
스키마(키/섹션/줄바꿈 스타일)는 예시 그대로 따르고, 내용은 입력 맥락에 맞게 작성하되, 참고(ref)의 톤/구성을 존중하라.
응답은 **JSON 배열 한 개**. 마크다운/설명 금지.

[입력 요약]
- category: {p.get("category")}
- target: {p.get("target")}
- period_raw: {p.get("period_raw")}
- normalized_period(있으면 사용, 없으면 기간 문구 생략): {p.get("normalized_period")}
- products: {p.get("products")}
- brand: {p.get("brand")}
- insight: {p.get("insight")}
- optional: title_suggestion={p.get("title_suggestion")}, theme_labels={p.get("theme_labels")}

[출력 스키마 예시]
{json.dumps(example, ensure_ascii=False, indent=2)}

[참고 스타일 힌트(존중하되, 입력과 충돌 시 입력 우선)]
- ref_title: {ref_title}
- ref_main_banner: {ref_main}
- ref_coupon_section: {ref_coupon}
- ref_product_section: {ref_product}
- ref_event_notes: {ref_notes}

[작성 원칙]
- title: ref_title이 유효하면 그 톤을 계승해 작성. 없으면 **반드시** "{primary_brand} 브랜드 프로모션"으로 **정확히** 작성(다른 단어 추가 금지).
- 절대 계절/카테고리형 제목을 쓰지 말 것.
- 기간: normalized_period가 있으면 그 값을 그대로 표기. 없으면 기간 문구를 임의 생성하지 말고 생략.
- coupon_section, product_section, event_notes는 예시 서식(리스트 점/굵은 텍스트/이미지 플레이스홀더[])을 유지하되, 입력/트렌드를 반영해 자연스럽게 재작성.
""".strip()


def _category_prompt(parsed: Dict[str, Any], example: List[Dict[str, Any]], ref: Dict[str, Any] | None = None) -> str:
    p = parsed
    products = p.get("products") or []
    brands = p.get("brand") or []
    ex_keys = list(example[0].keys())
    n = sum(1 for k in ex_keys if k.startswith("product_section"))
    ref = ref or {}
    ref_title = ref.get("title","")
    ref_banner = ref.get("main_banner","")
    ref_sections = [s for s in (ref.get("sections") or []) if s][:n]  # 섹션 수만큼

    return f"""
너는 올리브영 카테고리/계절 프로모션 JSON을 작성하는 카피라이터다.
스키마(키/섹션/줄바꿈 스타일)는 예시 그대로 따르고, 내용은 입력 맥락에 맞게 작성하라.
응답은 **JSON 배열 한 개**만 출력한다. 마크다운/설명 금지.

[입력 요약]
- category: {p.get("category")}
- target: {p.get("target")}
- period_raw: {p.get("period_raw")}
- normalized_period(있으면 사용, 없으면 날짜를 쓰지 말 것): {p.get("normalized_period")}
- products: {products}
- brands: {brands}
- insight: {p.get("insight")}
- optional: title_suggestion={p.get("title_suggestion")}, theme_labels={p.get("theme_labels")}

[출력 스키마 예시]
{json.dumps(example, ensure_ascii=False, indent=2)}

[참고 스타일 힌트(존중하되, 입력과 충돌 시 입력 우선)]
- ref_title: {ref_title}
- ref_main_banner: {ref_banner}
- ref_sections(앞에서부터 섹션별 힌트): {json.dumps(ref_sections, ensure_ascii=False)}

[작성 원칙]
- title: **카테고리/시즌 기반 제목**으로. ref_title이 유효하면 톤/표현을 참고하라.
- 섹션 수: 예시에 있는 product_section1~{n} **정확히 {n}개**만 생성한다. 빈 섹션 금지.
- 섹션명/카피: 입력 카테고리/상품/트렌드를 바탕으로 자연스럽게 구성하되, ref_sections의 톤/구조를 참고해 재작성해도 좋다.
- 상품 표기: `1. [브랜드 상품 이미지]` 한 줄씩. 브랜드 다수면 가장 어울리는 1개를 붙인다(새 브랜드 생성 금지).
- 기간: normalized_period가 있으면 그대로, 없으면 생략.
- 예시 서식(리스트 점/굵은 텍스트/이미지 플레이스홀더[]) 유지.
""".strip()


# ===== Node ======
def get_llm():
    key = settings.GOOGLE_API_KEY
    if not key:
        raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 설정되지 않았습니다.")
    return ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=key,
    temperature=0.3,
    response_mime_type="application/json"
)

def llm_decide_plan_type(parsed: Dict[str, Any]) -> str:
    llm = get_llm()
    prompt = LLM_DECIDE_PLAN_PROMPT.format(
        category=parsed.get("category"),
        target=parsed.get("target"),
        products=parsed.get("products"),
        brand=parsed.get("brand"),
        theme_labels=parsed.get("theme_labels"),
        insight=(parsed.get("insight") or "")[:800]
    )
    raw = llm.invoke(prompt).content
    raw = _strip_code_fence(raw)
    try:
        choice = json.loads(raw).get("plan_type", "")
        if choice in ("단일 프로모션", "카테고리/계절 프로모션"):
            return choice
    except Exception:
        pass
    # 실패 시 기본값(보수적)
    return "단일 프로모션"

def llm_parse_demo_insight(block: str) -> Dict[str, Any]:
    llm = get_llm()
    raw = llm.invoke(LLM_PARSE_PROMPT.replace("{TEXT}", block)).content
    raw = _strip_code_fence(raw)

    try:
        data = json.loads(raw)
    except Exception:
        # JSON 파싱 실패 시 안전값
        data = {}

    parsed = {
        "category":          _as_list(data.get("category")),
        "target":            _as_list(data.get("target")),
        "period_raw":        _as_list(data.get("period_raw")),
        "products":          _as_list(data.get("products")),
        "brand":             _as_list(data.get("brand")),
        "insight":           (data.get("insight") or "").strip(),
        "normalized_period": (data.get("normalized_period") or "").strip(),
        "title_suggestion":  (data.get("title_suggestion") or "").strip(),
        "theme_labels":      _as_list(data.get("theme_labels")),
        "confidence":        data.get("confidence") or {}
    }
    return parsed

def fetch_exam_example(plan_type: str) -> Optional[List[Dict[str, Any]]]:
    try:
        url = settings.FORMATTER_SUPERBASE_URL
        key = settings.FORMATTER_SUPABASE_ANON_KEY
        supabase: Client = create_client(url, key)
        resp = supabase.rpc("get_exam_by_plan_type", {"requested_plan_type": plan_type}).execute()
        if resp.data:
            ex = resp.data[0].get("exam")
            if isinstance(ex, list) and ex:
                return ex
        return None
    except Exception:
        return None


def node_llm_parse(state: PlanState) -> PlanState:
    try:
        parsed = llm_parse_demo_insight(state["demo_insight"])
        state["parsed"] = parsed
    except Exception as e:
        state["error"] = f"LLM parse error: {e}"
    return state

def node_decide_plan(state: PlanState) -> PlanState:
    if state.get("plan_type") and state["plan_type"] != "auto":
        return state
    try:
        state["plan_type"] = llm_decide_plan_type(state.get("parsed", {}) or {})
    except Exception:
        state["plan_type"] = "단일 프로모션"
    return state
DESIRED_ORDER_SINGLE = ["title","main_banner","coupon_section","product_section","event_notes"]
DESIRED_ORDER_CATEGORY = ["title","main_banner","product_section1","product_section2","product_section3"]

def reorder_dict(d: dict, order: List[str]) -> dict:
    # 지정 순서대로 먼저 배치
    out = {k: d[k] for k in order if k in d}
    # 남은 키는 뒤에 유지
    for k, v in d.items():
        if k not in out:
            out[k] = v
    return out

def node_fetch_schema_hint(state: PlanState) -> PlanState:
    supa_ex = fetch_exam_example(state["plan_type"])  # 원격 예시 시도
    parsed = state.get("parsed", {}) or {}

    if state["plan_type"] == "단일 프로모션":
        supa = (supa_ex or [{}])[0]
        primary_brand = (parsed.get("brand") or [""])[0]

        ex = {
            "title": supa.get("title", f"{primary_brand} 브랜드 프로모션"),
            "main_banner": supa.get("main_banner", ""),
            "coupon_section": supa.get("coupon_section", ""),
            "product_section": supa.get("product_section", ""),
            "event_notes": supa.get("event_notes", ""),
        }
        # 키 순서 고정
        order = ["title","main_banner","coupon_section","product_section","event_notes"]
        ex = {k: ex[k] for k in order}

        state["schema_hint"] = {
            "example": [ex],
            "ref": {
                "title": supa.get("title",""),
                "main_banner": supa.get("main_banner",""),
                "coupon_section": supa.get("coupon_section",""),
                "product_section": supa.get("product_section",""),
                "event_notes": supa.get("event_notes",""),
            }
        }
        return state

    # 카테고리/계절
    n = max(1, min(len(parsed.get("products") or []), 5))
    supa = (supa_ex or [{}])[0]
    ex = {"title": supa.get("title",""), "main_banner": supa.get("main_banner","")}
    for i in range(1, n+1):
        key = f"product_section{i}"
        ex[key] = supa.get(key, "")
    # 키 순서 고정
    section_keys = [k for k in ex if k.startswith("product_section")]
    order = ["title","main_banner"] + section_keys
    ex_ordered = {k: ex[k] for k in order}

    state["schema_hint"] = {
        "example": [ex_ordered],
        "ref": {
            "title": supa.get("title",""),
            "main_banner": supa.get("main_banner",""),
            "sections": [supa.get(f"product_section{i}","") for i in range(1, n+1)]
        }
    }
    return state


