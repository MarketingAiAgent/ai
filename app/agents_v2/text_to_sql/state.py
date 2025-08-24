from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field 

class SQLState(BaseModel): 
    # 주어진 값
    question: str = Field(description="사용자 질문")
    schema_info: str = Field(description="데이터베이스 스키마 정보")
    conn_str: str = Field(description="데이터베이스 연결 문자열")
    graph_type: str = Field(default="", description="그래프 타입")
    
    # 루프 로직 
    tried: int = Field(default=0, description="시도 횟수")
    error: Optional[str] = Field(default=None, description="에러 메시지")

    # 만드는 값
    query: Optional[Any] = Field(default=None, description="생성된 SQL 쿼리")
    data_json: Optional[Dict[str, Any]] = Field(default=None, description="쿼리 결과 데이터")
    graph_json: Optional[str] = Field(default=None, description="그래프 JSON")