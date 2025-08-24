from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Literal

# ===== Main State =====
class AgentState(BaseModel):
    history: List[Dict[str, str]] = []
    user_message: str = ""
    intent: Literal["QA", "Promotion", "Out-of-scope"]

    def __init__(self, history: List[Dict[str, str]], user_message: str):
        self.history = history
        self.user_message = user_message
        self.intent = "Out-of-scope"

# ===== Promotion ======

Scope = Literal["brand", "category"] 

class PromotionSlots(BaseModel):
    
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # Optional
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

    # Required (논리상 필수; 진행 결정에 사용)
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

    # ---- 정규화 ----
    @field_validator("audience", "KPI", "concept", "period", "target", mode="before")
    @classmethod
    def _strip_empty(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("scope", mode="before")
    @classmethod
    def _normalize_scope(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if s in ("브랜드", "제품"):
            return s
        # 유사 표기 보정(과도한 보정은 지양)
        if "브랜드" in s:
            return "브랜드"
        if "제품" in s:
            return "제품"
        return None

    # ---- 편의 메서드 ----
    def merge_missing(self, other: "PromotionSlots") -> "PromotionSlots":
        """
        자기 자신의 값은 유지하고, 비어 있는 필드만 other에서 채웁니다.
        (보수적 병합; 확정값 덮어쓰지 않음)
        """
        return PromotionSlots(
            audience=self.audience or other.audience,
            KPI=self.KPI or other.KPI,
            concept=self.concept or other.concept,
            scope=self.scope or other.scope,
            period=self.period or other.period,
            target=self.target or other.target,
        )

    def missing_flags(self) -> "MissingFlags":
        """
        다음 질문/행동 결정을 위한 결측 플래그를 계산합니다.
        - 규칙:
          1) scope 또는 period가 비어 있으면 최우선으로 물어봅니다.
          2) scope/period가 채워졌고 target이 비어 있으면 옵션 소싱을 통해 묻습니다.
        """
        need_scope_period = (self.scope is None) or (self.period is None)
        need_target = (not need_scope_period) and (self.target is None)
        return MissingFlags(
            need_scope_period=need_scope_period,
            need_target=need_target,
        )

    def decide_next_action(self) -> "NextAction":
        flags = self.missing_flags()
        if flags.need_scope_period:
            return NextAction.ASK_SCOPE_PERIOD
        if flags.need_target:
            return NextAction.ASK_TARGET_WITH_OPTIONS
        # scope/period/target이 모두 채워진 경우
        return NextAction.RECAP_CONFIRM
