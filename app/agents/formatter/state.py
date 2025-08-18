from typing import TypedDict, List, Dict, Any, Optional

class PlanState(TypedDict, total=False):
    plan_type: str                      # "단일 프로모션" | "카테고리/계절 프로모션" | "auto"
    demo_insight: str                   # 원문 입력(형식 제약 없음)
    parsed: Dict[str, Any]              # LLM 파싱 결과
    schema_hint: Dict[str, Any]         # 스키마 예시(원격/로컬)
    draft_exam: List[Dict[str, Any]]    # LLM 생성 초안
    final_exam: List[Dict[str, Any]]    # 검증 통과 최종본
    error: Optional[str]