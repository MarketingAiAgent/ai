from typing import List, Dict, Optional 
from pydantic import BaseModel, Field 

class SQLState(BaseModel): 
    # 주어진 값
    question: str
    schema_info: str 
    conn_str: str
    graph_type: str 

    # 루프 로직 
    tried: int = 0
    error: Optional[str] = None

    # 만드는 값
    query: str = "" 
    data_json: Optional[str] = None
    graph_json: Optional[str] = None