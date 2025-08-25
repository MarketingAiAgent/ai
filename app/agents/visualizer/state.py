from pydantic import BaseModel, Field
from typing import Optional, Any

class VisualizeState(BaseModel):
    # 입력
    user_question: str = Field(description="사용자 질문")
    instruction: str = Field(description="시각화 지시사항")
    json_data: Optional[Any] = Field(default=None, description="시각화할 JSON 데이터")

    # 결과
    json_graph: str = Field(default="", description="생성된 그래프 JSON")
    output: str = Field(default="", description="출력 텍스트")
    error: str = Field(default="", description="에러 메시지")