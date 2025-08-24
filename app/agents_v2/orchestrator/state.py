from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Literal, Optional, Any

from app.agents_v2.orchestrator.nodes.qa_plan import QAPlan
from app.agents_v2.orchestrator.nodes.plan_options import OptionToolPlans

# ===== Promotion ======

Scope = Literal["brand", "category"] 

class PromotionSlots(BaseModel):
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    audience: Optional[str] = Field(
        default=None,
        description="예: '20대', '대학생'. 사용자가 먼저 말한 경우에만 채움.",
    )
    KPI: Optional[str] = Field(
        default=None,
        description="예: 'CTR 2%+', '신규구매 500건'. 사용자가 먼저 말한 경우에만 채움.",
    )
    concept: Optional[str] = Field(
        default=None,
        description="예: '신학기 번들', '크리에이터 협업'. 옵션 소싱 중 제안되면 채움.",
    )   

    scope: Optional[Scope] = Field(
        default=None,
        description="브랜드 | 제품",
    )
    period: Optional[str] = Field(
        default=None,
        description="예: '다음 달 4주', '이번 달 2주'.",
    )
    target: Optional[str] = Field(
        default=None,
        description="scope가 브랜드면 브랜드명, 제품이면 제품명.",
    )

    # ---- 편의 메서드 ----
    def merge_missing(self, other) -> "PromotionSlots":
        if isinstance(other, dict):
            other_audience = other.get("audience")
            other_KPI = other.get("KPI")
            other_concept = other.get("concept")
            other_scope = other.get("scope")
            other_period = other.get("period")
            other_target = other.get("target")
        else:
            other_audience = other.audience
            other_KPI = other.KPI
            other_concept = other.concept
            other_scope = other.scope
            other_period = other.period
            other_target = other.target
            
        return PromotionSlots(
            audience=self.audience or other_audience,
            KPI=self.KPI or other_KPI,
            concept=self.concept or other_concept,
            scope=self.scope or other_scope,
            period=self.period or other_period,
            target=self.target or other_target,
        )

    def decide_next_action(self) -> Literal["ASK_SCOPE_PERIOD", "ASK_TARGET_WITH_OPTIONS", "RECAP_CONFIRM"]:
        if self.scope is None or self.period is None:
            return "ASK_SCOPE_PERIOD"
        if self.target is None:
            return "ASK_TARGET_WITH_OPTIONS"
        return "RECAP_CONFIRM"

# ===== Main State =====
class AgentState(BaseModel):
    model_config = ConfigDict(extra="allow")
    
    # 기본 정보
    history: List[Dict[str, str]] = Field(default_factory=list)
    user_message: str = Field(default="")
    intent: Literal["QA", "Promotion", "Out-of-scope"] = Field(default="Out-of-scope")
    
    # 프로모션 관련
    promotion_slots: Optional[PromotionSlots] = Field(default=None)
    tool_plans: Optional[OptionToolPlans] = Field(default=None)
    
    # QA 관련
    qa_plan: Optional[QAPlan] = Field(default=None)
    qa_table: Optional[Dict[str, Any]] = Field(default=None)
    qa_chart: Optional[str] = Field(default=None)
    qa_snapshot: Optional[Dict[str, Any]] = Field(default=None)
    qa_web_rows: Optional[List[Dict[str, Any]]] = Field(default=None)
    qa_explanation: Optional[str] = Field(default=None)
    
    # 응답 관련
    response: Optional[str] = Field(default=None)
    graph: Optional[Dict[str, Any]] = Field(default=None)
    table: Optional[Dict[str, Any]] = Field(default=None)
    snapshot: Optional[Dict[str, Any]] = Field(default=None)
    
    # 도구 실행 결과
    sql_rows: Optional[List[Dict[str, Any]]] = Field(default=None)
    web_rows: Optional[List[Dict[str, Any]]] = Field(default=None)
    
    # 분기 제어
    expect_fields: List[str] = Field(default_factory=list)
    options: Optional[List[Any]] = Field(default=None)
    
    # 리포트 관련
    report: Optional[Any] = Field(default=None)
    report_markdown: Optional[str] = Field(default=None)
    
    # 편의 메서드
    def promotion_slots_dict(self) -> Optional[Dict[str, Any]]:
        """하위 호환성을 위한 메서드"""
        return self.promotion_slots.model_dump() if self.promotion_slots else None