import pandas as pd 
import logging
from sqlalchemy import text, create_engine

from langgraph.graph import StateGraph, END

from app.core.config import settings 
from app.core.data_utils import normalize_table_data, create_empty_table
from .crew import crewAI_sql_generator
from .state import *

logger = logging.getLogger(__name__)
MAX_ROWS = 20

# --- Node --- 
def call_t2s_crew(state: SQLState): 
    message = state.question 
    if state.error is not None: 
        message += f"\n\n**주의점** 지난 번 생성한 SQL에서는 다음과 같은 에러가 발생했습니다: \n{state.error}\n 같은 실수를 반복하지 마세요."
    sql = crewAI_sql_generator(message=state.question, schema_info=state.schema_info)

    state.query = text(sql)
    state.error = None
    
    return state 

def call_sql(state: SQLState):
    engine = None
    try:
        engine = create_engine(state.conn_str, pool_pre_ping=True)

        # SQL 실행
        df = pd.read_sql_query(state.query, engine)

        # 공통 유틸리티로 데이터 정규화
        state.data_json = normalize_table_data(df, MAX_ROWS)

        logger.info(
            "SQL 실행 성공 | row_count=%s, columns=%s",
            state.data_json["row_count"],
            state.data_json["columns"],
        )

        logger.info(f"{state.query}")

    except Exception as e:
        # 실패해도 data_json은 동일 스키마로 채워서 downstream이 깨지지 않게
        state.data_json = create_empty_table(str(e))
        state.error = str(e)
        state.tried = getattr(state, "tried", 0) + 1
        logger.error(f"SQL 실행 실패:{e}")

    finally:
        # 커넥션 정리
        try:
            if engine is not None:
                engine.dispose()
        except Exception:
            pass

    return state
    
def check_table(state: SQLState): 
    if state.error is None and state.data_json and state.data_json.get("row_count", 0) > 0:
        return "next"
    elif state.tried < 3:  # 최대 3번 재시도
        return "redo"
    else:
        return "next"  # 실패해도 다음으로 진행

# --- Graph --- 
workflow = StateGraph(SQLState)

workflow.add_node('generate_sql', call_t2s_crew)
workflow.add_node('make_table', call_sql)

workflow.set_entry_point("generate_sql")
workflow.add_edge("generate_sql", "make_table")
workflow.add_conditional_edges("make_table", check_table, {'next': END, "redo": "generate_sql"})

t2s_app = workflow.compile()