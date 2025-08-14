import json
import textwrap
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langgraph.graph import StateGraph, END

from .state import *
from app.core.config import settings 
from app.agents.text_to_sql.__init__ import call_sql_generator

logger = logging.getLogger(__name__)

# --- 툴 함수 ---
def run_t2s_agent(state: OrchestratorState):
    instruction = state['instructions'].t2s_instruction

    logger.info("T2S 에이전트 실행: %s", instruction)

    state = call_sql_generator(message=instruction, conn_str=state['conn_str'], schema_info=state['schema_info'])
    table = state.data_json

    if isinstance(table, str): 
        table = json.loads(table)

    return table

# 일단 모조품만
def run_knowledge_agent(instruction: str) -> str:
    logger.info("지식 에이전트 실행: %s", instruction)
    return "최근 숏폼 콘텐츠를 활용한 바이럴 마케팅이 인기입니다."

# --- LangGraph 노드 정의 ---
def planner_node(state: OrchestratorState):
    """실행 계획을 수립합니다."""

    logger.info("--- 1. 🤔 계획 수립 노드 (Planner) 실행 ---")

    parser = PydanticOutputParser(pydantic_object=OrchestratorInstruction)
    
    prompt_template = textwrap.dedent("""
    You are a master AI orchestrator. Your job is to analyze the user's request and the current conversation state, then create a complete, step-by-step plan for this turn.
    The plan must be a JSON object that follows this format: {format_instructions}

    # Rules:
    1.  **Analyze Task:** First, check if there is an `active_task` for 'promotion'. If not, and the user wants to start one, create a new task.
    2.  **Plan Tool Use:** Based on the user's message and the current `slots`, decide if you need to call tools (`t2s_agent` or `knowledge_agent`).
    3.  **Plan Response:** Based on the plan, create an instruction for the `response_generator`.

    # Current State & History:
    - User Message: "{user_message}"
    - Active Task: {active_task}
    - History: {history}
    """)
    
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, model_kwargs={"response_format": {"type": "json_object"}}, api_key=settings.GOOGLE_API_KEY)
    chain = prompt | llm | parser

    instructions = chain.invoke({
        "user_message": state['user_message'],
        "active_task": state['active_task'].model_dump_json() if state.get('active_task') else 'None',
        "history": json.dumps(state['history'])
    })
    
    return {"instructions": instructions}

def tool_executor_node(state: OrchestratorState):
    """필요한 툴을 병렬로 실행합니다."""
    logger.info("--- 2. 🔨 툴 실행 노드 (Tool Executor) 실행 ---")
    
    instructions = state.get("instructions")
    if not instructions or (not instructions.t2s_instruction and not instructions.knowledge_instruction):
        logger.info("호출할 툴이 없습니다.")
        return {"tool_results": None}

    tool_results = {}
    
    with ThreadPoolExecutor() as executor:
        futures = {}
        if instructions.t2s_instruction:
            futures[executor.submit(run_t2s_agent, instructions.t2s_instruction)] = "t2s"
        if instructions.knowledge_instruction:
            futures[executor.submit(run_knowledge_agent, instructions.knowledge_instruction)] = "knowledge"
        
        for future in futures:
            tool_name = futures[future]
            try:
                tool_results[tool_name] = future.result()
            except Exception as e:
                logger.error("%s 툴 실행 중 에러 발생: %s", tool_name, e)
                tool_results[tool_name] = f"Error: {e}"
    
    return {"tool_results": tool_results}

def response_generator_node(state: OrchestratorState):
    """최종 응답을 생성하고 히스토리를 업데이트합니다."""
    logger.info("--- 3. 🗣️ 응답 생성 노드 (Response Generator) 실행 ---")
    
    instructions = state.get("instructions")
    tool_results = state.get("tool_results")
    
    response_instruction = instructions.response_generator_instruction
    final_response = f"[응답 생성 지시: {response_instruction}]"
    
    if tool_results:
        final_response += f"\n\n[툴 실행 결과: {json.dumps(tool_results, ensure_ascii=False)}]"

    history = state.get("history", [])
    history.append({"role": "user", "content": state["user_message"]})
    history.append({"role": "assistant", "content": final_response})
    
    return {"history": history, "user_message": ""}

# --- 그래프 구성 및 컴파일 ---
workflow = StateGraph(OrchestratorState)

workflow.add_node("planner", planner_node)
workflow.add_node("tool_executor", tool_executor_node)
workflow.add_node("response_generator", response_generator_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "tool_executor")
workflow.add_edge("tool_executor", "response_generator")
workflow.add_edge("response_generator", END)

orchestrator_app = workflow.compile()