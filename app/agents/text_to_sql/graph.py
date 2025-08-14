import pandas as pd 
import logging
from sqlalchemy import text, create_engine

from langgraph.graph import StateGraph, END

from app.core.config import settings 
from .crew import crewAI_sql_generator
from .state import *

logger = logging.getLogger(__name__)

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
    try: 
        engine = create_engine(state.conn_str)
        query_plan_df = pd.read_sql_query(state.query, engine)
        state.data_json = query_plan_df.to_json()
        logger.info("SQL 생성 성공")
    except Exception as e: 
        state.error = e
        state.tried += 1
        logger.error(f"SQL 생성 실패: {e}")
        
    return state 

def check_table(state: SQLState): 
    if state.error is None | state.tried > 2: 
        return "next"
    else: 
        return "redo"

# --- Graph --- 
workflow = StateGraph(SQLState)

workflow.add_node('generate_sql', call_t2s_crew)
workflow.add_node('make_table', call_sql)

workflow.set_entry_point("generate_sql")
workflow.add_edge("generate_sql", "make_table")
workflow.add_conditional_edges("make_table", check_table, {'next': END, "redo": "generate_sql"})

t2s_app = workflow.compile()