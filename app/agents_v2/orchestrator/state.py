from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Literal, Optional, Any
import uuid

# ===== QA 관련 State =====

Choice = Literal["t2s", "web", "both", "none"]
AllowedSources = Literal["supabase_marketing", "supabase_beauty", "tavily"]

class PlanningNote(BaseModel):
    """스키마/컬럼명을 모른다는 전제의 CoT 요약(구조화)."""
    user_intent: str
    needed_table: str                  # 예: "행=요일, 지표=전환 관련 핵심 요약"
    filters: List[str] = Field(default_factory=list)        # 기간/세그먼트 등(스키마 명시 금지)
    granularity: Optional[str] = None  # 일/주/월, 캠페인/채널 등
    comparison: Optional[str] = None   # 전년동기/직전기간 등
    why_t2s: Optional[str] = None
    why_web: Optional[str] = None

class T2SPlan(BaseModel):
    enabled: bool = True
    instruction: Optional[str] = None  
    top_rows: int = 100
    visualize: bool = True
    viz_hint: Optional[str] = None     

class WebPlan(BaseModel):
    enabled: bool = True
    query: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    use_sources: List[AllowedSources] = Field(
        default_factory=lambda: ["supabase_marketing", "supabase_beauty", "tavily"]
    )
    top_k: int = 5
    scrape_k: int = 0

class QAPlan(BaseModel):
    choice: Choice
    planning: PlanningNote
    t2s: T2SPlan
    web: WebPlan

# ===== 프로모션 옵션 관련 State =====

ToolChoice = Literal["sql", "web", "both", "none"]

class OptionPlanningNote(BaseModel):
    goal: str = Field(description="이번 턴의 목적(예: 20대 대상, 브랜드 기준 타겟 후보 상위 3개)")
    needed_table: str = Field(description="원하는 표의 의미/행 단위 정의(브랜드/제품 등)와 집계 기준")
    filters: List[str] = Field(default_factory=list, description="기간·오디언스·기타 필터 조건(스키마명 지정 금지)")
    metrics_preference: List[str] = Field(default_factory=list, description="선호 지표의 개념적 우선순위(스키마에 없으면 대체 허용)")
    ranking_logic: str = Field(description="정렬/상위 선정의 개념적 기준(스키마 기반 자유 선택)")
    notes: Optional[str] = None

class SQLPlan(BaseModel):
    enabled: bool = True
    instruction: Optional[str] = None       # 최종 자연어 인스트럭션(우선)
    queries: List[str] = Field(default_factory=list)  # 백업 후보(선택)
    top_k: int = 3

class OptionWebPlan(BaseModel):
    enabled: bool = True
    query: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    use_sources: List[AllowedSources] = Field(
        default_factory=lambda: ["supabase_marketing", "supabase_beauty", "tavily"]
    )
    top_k: int = 3
    scrape_k: int = 0

class OptionToolPlans(BaseModel):
    tool_choice: ToolChoice
    planning: OptionPlanningNote
    sql: SQLPlan
    web: OptionWebPlan

# ===== 타겟 옵션 관련 State =====

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

# ===== 프로모션 리포트 관련 State =====

class ReportHighlights(BaseModel):
    title: str
    summary: str
    slots_recap: Dict[str, str]
    highlights: List[str]
    markdown: str
    plan: Dict[str, Any] = Field(default_factory=dict)
    kpis: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)

class ReportNodeOutput(BaseModel):
    message: str
    report: ReportHighlights

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

    def decide_next_action(self) -> tuple[Literal["ASK_SCOPE_PERIOD", "ASK_TARGET_WITH_OPTIONS", "RECAP_CONFIRM"], list[str]]:
        """다음 액션과 필요한 필드들을 반환합니다."""
        if self.scope is None or self.period is None:
            missing = []
            if self.scope is None:
                missing.append("scope")
            if self.period is None:
                missing.append("period")
            return "ASK_SCOPE_PERIOD", missing
        if self.target is None:
            return "ASK_TARGET_WITH_OPTIONS", ["target"]
        return "RECAP_CONFIRM", []

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
    
    # 실행 컨텍스트
    sql_context: Optional[Dict[str, Any]] = Field(default=None)
    
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