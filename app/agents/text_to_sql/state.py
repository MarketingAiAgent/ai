from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field 

class SQLState(BaseModel): 
    # 주어진 값
    question: str
    schema_info: str 
    conn_str: str
    graph_type: str = ""

    # 루프 로직 
    tried: int = 0
    error: Optional[str] = None

    # 만드는 값
    query: Optional[Any] = None
    data_json: Optional[Any] = None
    graph_json: Optional[str] = None
    dataframe: Optional[Any] = None  # DataFrame 객체 (export용)