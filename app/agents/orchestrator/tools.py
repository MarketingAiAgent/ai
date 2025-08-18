import logging 
import json 

from app.agents.text_to_sql.__init__ import call_sql_generator
from .state import *

logger = logging.getLogger(__name__)

def run_t2s_agent(state: OrchestratorState):
    instruction = state['instructions'].t2s_instruction
    logger.info("T2S 에이전트 실행: %s", instruction)

    result = call_sql_generator(message=instruction, conn_str=state['conn_str'], schema_info=state['schema_info'])
    sql = result['query']
    table = result["data_json"]

    logger.info(f"쿼리: \n{sql}")
    logger.info(f"결과 테이블: \n{table}")

    if isinstance(table, str):
        table = json.loads(table)
    return table

def run_knowledge_agent(instruction: str) -> str:
    logger.info("지식 에이전트 실행: %s", instruction)
    return "최근 숏폼 콘텐츠를 활용한 바이럴 마케팅이 인기입니다."
