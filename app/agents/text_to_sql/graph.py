import pandas as pd 
import logging
from sqlalchemy import text, create_engine
from pandas.api.types import is_datetime64_any_dtype

from langgraph.graph import StateGraph, END

from app.core.config import settings 
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

        # 실행
        df = pd.read_sql_query(state.query, engine)

        # 멀티컬럼 방어
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["__".join(map(str, c)).strip() for c in df.columns.values]

        # 날짜 컬럼 ISO 문자열화
        for col in df.columns:
            try:
                if is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                pass

        # NaN/NaT -> None (JSON null)
        df = df.where(pd.notnull(df), None)

        # 미리보기만 담고 전체 행수는 별도 기입
        preview = df.head(MAX_ROWS)

        state.data_json = {
            "rows": preview.to_dict(orient="records"),     # ✅ [{col:val}, ...]
            "columns": [str(c) for c in preview.columns],  # ✅ 열 이름
            "row_count": int(df.shape[0]),                 # ✅ 전체 행 수
        }

        logger.info(
            "SQL 실행 성공 | row_count=%s, columns=%s",
            state.data_json["row_count"],
            state.data_json["columns"],
        )

        logger.info(f"{state.query}")

    except Exception as e:
        # 실패해도 data_json은 동일 스키마로 채워서 downstream이 깨지지 않게
        state.data_json = {"rows": [], "columns": [], "row_count": 0, "error": str(e)}
        state.error = e
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
    if state.error is None or  state.tried > 2: 
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