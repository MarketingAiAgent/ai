from pydantic import BaseModel
from typing import Optional, Any

class VisualizeState(BaseModel):
    user_question: str
    instruction: str
    json_data: Optional[Any]

    # 결과
    json_graph: str = ""
    output: str = ""
    error: str = ""