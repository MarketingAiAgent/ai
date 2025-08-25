import re 
from crewai import Agent, Crew, Task, Process, LLM
from app.core.config import settings 

def crewAI_sql_generator(message, schema_info, LLM_MODEL="gemini/gemini-2.5-flash"):
    llm = LLM(
        model=LLM_MODEL,
        temperature=0.0,
        api_key=settings.GOOGLE_API_KEY
    )

    query_parser = Agent(
        role="QueryParserAgent",
        goal="사용자의 자연어 질문을 SQL 분석 명세로 구조화한다: '{input}'",
        backstory=(
            "당신은 마케팅/커머스 데이터 분석 전문가입니다. "
            "질문 속 인구통계, 시간, 지역, 행동 필터 조건을 정확히 파악하고, "
            "지표(예: 구매 수, 매출 합계 등)를 기반으로 JSON 명세로 구조화하세요. "
            "특히 '누가', '무엇을', '얼마나', '언제' 등을 놓치지 마세요."
            f"주어진 데이터베이스 스키마 정보를 반드시 참고해야 합니다.\n\n--- 스키마 정보 ---\n{schema_info}"
        ),
        verbose=False,
        llm=llm
    )

    advanced_sql_agent = Agent(
        role="AdvancedSQLAgent",
        goal="분석 명세와 스키마를 바탕으로, 스스로 검토하고 개선하여 최종적으로 정확한 SQL 쿼리를 생성한다.",
        backstory=(
            "당신은 SQL 작성, 검토, 최적화를 한 번의 흐름으로 수행하는 최고의 데이터 분석가입니다. "
            "초안을 만든 후, 논리적 오류나 비효율성이 없는지 스스로 검증하고 최종 쿼리를 제시합니다. "
            f"주어진 데이터베이스 스키마 정보를 반드시 참고해야 합니다.\n\n--- 스키마 정보 ---\n{schema_info}"
        ),
        verbose=False,
        llm=llm
    )

    task_parser = Task(
        description=(
            "사용자 질문에 포함된 대상(무엇을), 조건(누가, 언제, 어디서), 지표(얼마나)를 모두 식별하여 "
            "분석 가능한 JSON 명세로 구조화하세요."
        ),
        agent=query_parser,
        expected_output="분석의 핵심 요소를 담은 JSON 형식의 명세서."
    )

    task_advanced_sql = Task(
        description=(
            "앞 단계에서 생성된 JSON 명세를 바탕으로 최종 SQL 쿼리를 생성하라. "
            "생성 과정에서 반드시 쿼리의 정확성과 효율성을 스스로 검토하고 개선하는 과정을 포함해야 한다."
        ),
        agent=advanced_sql_agent,
        context=[task_parser], 
        expected_output="최종적으로 개선된 SQL 쿼리. 설명 없이 ```sql ... ``` 형식으로만 출력."
    )
    crew = Crew(
        agents=[query_parser, advanced_sql_agent],
        tasks=[task_parser, task_advanced_sql],
        process=Process.sequential,
        verbose=False,
        cache=True
    )

    result = crew.kickoff(inputs={"input": message})
    sql_output = result.tasks_output[1].raw

    match = re.search(r"```sql\s*(.*?)```", sql_output, re.DOTALL)
    if match:
        sql_query = match.group(1).strip()
    else:
        sql_query = sql_output.strip()

    return sql_query