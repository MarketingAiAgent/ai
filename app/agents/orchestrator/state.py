from typing import Literal, TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field

# --- Pydantic 모델 (데이터 검증 및 구조화용) ---
class PromotionSlots(BaseModel):
    objective: Optional[str] = Field(None, description="사용자가 정의한 이 프로모션으로 이루고자 하는 것")
    target_type: Optional[Literal["brand_target", "category_target"]] = Field(None, description="프로모션의 타겟 종류")
    target: Optional[str] = Field(None, description="프로모션 타겟 고객")
    product_options: List[str] = Field(default_factory=list, description="사용자에게 제안할 상품 후보 목록")
    selected_product: Optional[str] = Field(None, description="사용자가 최종 선택한 프로모션 상품")
    brand: Optional[str] = Field(None, description="사용자가 선택한 프로모션 타겟 브랜드")
    duration: Optional[str] = Field(None, description="프로모션 기간")

class ActiveTask(BaseModel):
    task_id: str
    status: Literal["in_progress", "done"] # 쓸지는 모르겠지만 일단 두자
    slots: Optional[PromotionSlots] = Field(None, description="task_type이 promotion일 경우 진행 상황")

class OrchestratorInstruction(BaseModel):
    t2s_instruction: Optional[str] = Field(None, description="만약 t2s 에이전트 호출이 없다면 None, 호출이 있다면 t2s 에이전트가 해야할 일을 지시")
    knowledge_instruction: Optional[str] = Field(None, description="만약 지식 에이전트 호출이 없다면 None, 호출이 있다면 지식 에이전트가 해야할 일을 지시")
    response_generator_instruction: str = Field(description="응답 생성 에이전트가 어떤 응답을 해야하는지 지시")
    
# --- LangGraph의 상태 (State) ---
class OrchestratorState(TypedDict):
    # --- 이전 Context --- 
    history: List[Dict[str, str]]
    active_task: Optional[ActiveTask]

    # --- DB 관련 ---
    schema_info: str 
    conn_str: str

    # --- 그래프 관련 --- 
    user_message: str
    instructions: Optional[OrchestratorInstruction] = None
    tool_results: Optional[Dict[str, Any]] = None
    output: str = ""

# --- initial_state 생성 함수 --- 
def return_initial_state(history, active_task, conn_str, schema_info,message):
    
    return OrchestratorState(
        history=history,
        active_task=active_task,
        schema_info=schema_info,
        conn_str=conn_str,
        user_message=message
    )