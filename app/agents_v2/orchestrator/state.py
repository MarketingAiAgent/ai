from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Literal, Optional, Any

from app.agents_v2.orchestrator.nodes.qa_plan import QAPlan

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
    def merge_missing(self, other: "PromotionSlots") -> "PromotionSlots":
        return PromotionSlots(
            audience=self.audience or other.audience,
            KPI=self.KPI or other.KPI,
            concept=self.concept or other.concept,
            scope=self.scope or other.scope,
            period=self.period or other.period,
            target=self.target or other.target,
        )

    def decide_next_action(self) -> Literal["ASK_SCOPE_PERIOD", "ASK_TARGET_WITH_OPTIONS", "RECAP_CONFIRM"]:
        if self.scope is None or self.period is None:
            return "ASK_SCOPE_PERIOD"
        if self.target is None:
            return "ASK_TARGET_WITH_OPTIONS"
        return "RECAP_CONFIRM"

# ===== Main State =====
class AgentState(BaseModel):
    history: List[Dict[str, str]] = []
    user_message: str = ""
    intent: Literal["QA", "Promotion", "Out-of-scope"]
    promotion_slots: Optional[PromotionSlots] = None
    response: Optional[str] = None
    qa_plan: Optional[QAPlan] = None
    graph: Optional[Dict[str, Any]] = None
    table: Optional[Dict[str, Any]] = None
    snapshot: Optional[Dict[str, Any]] = None
    tool_plan

    def __init__(self, history: List[Dict[str, str]], user_message: str, promotion_slots: Optional[PromotionSlots] = None):
        self.history = history
        self.user_message = user_message
        self.intent = "Out-of-scope"
        self.promotion_slots = promotion_slots
        self.response = None
        self.graph = None