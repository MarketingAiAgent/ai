import pandas as pd 
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import json 
import traceback
from uuid import uuid4
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types as genai_types
from typing import Optional

from .state import VisualizeState
from app.core.config import settings

# ===== Helper =====
class GeminiClient:
    def __init__(self, model: str):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model = model

    def generate(self, prompt: str, system_instruction: Optional[str] = None,
                 json_mode: bool = False, response_schema=None, max_output_tokens: int = 2048) -> str:
        cfg = genai_types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json" if json_mode else None,
            **({"response_schema": response_schema} if response_schema else {})
        )
        full_prompt = (system_instruction + "\n\n" + prompt) if system_instruction else prompt
        contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=full_prompt)])]
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
        return resp.text


def safe_exec(code: str, env: dict):
    try:
        compiled = compile(code, "<generated>", "exec")
    except SyntaxError as e:
        tb = "".join(traceback.format_exception_only(e.__class__, e))
        raise RuntimeError(f"⚠️ LLM 코드 SyntaxError:\n{tb}\n-----\n{code[:400]}")
    exec(compiled, env)

# ===== Prompt =====
VISUALIZER_PROMPT = ('''
아래는 사용자의 원래 질문과 SQL 쿼리 실행 결과의 JSON 데이터입니다.
이 정보를 바탕으로 파이썬의 **Plotly Express** 라이브러리를 사용하여 가장 적합한 시각화 코드를 생성해주세요.

**사용자의 원래 질문:**
{question}

**쿼리 결과 (JSON 데이터):**
{json_data}

**참고: 데이터 형태**
- 전달되는 데이터는 {{"rows": [...], "columns": [...], "row_count": int}} 구조입니다
- 실제 차트 데이터는 `data["rows"]`에 있으며, 각 행은 딕셔너리 형태입니다
- `data["columns"]`에는 컬럼명 리스트가 있습니다

**시각화 코드 생성 요구사항:**
1. **반드시 다음 구조로 코드를 작성하세요:**
   ```python
   # 데이터 구조 확인 및 추출
   if isinstance(data, dict) and "rows" in data:
       df = pd.DataFrame(data["rows"])
   else:
       df = pd.DataFrame(data)
   
   # 차트 생성
   fig = px.[chart_type](df, x='컬럼명', y='컬럼명', title='차트 제목')
   # 추가 설정 (필요시)
   ```

2. **차트 선택 가이드라인:**
   - 시계열 데이터: px.line 사용
   - 카테고리별 비교: px.bar 사용  
   - 분포 확인: px.histogram 사용
   - 상관관계: px.scatter 사용

3. **필수 규칙:**
   - print 함수 사용 금지
   - 파일 저장 코드 금지 (로컬 저장 안함)
   - 데이터는 그대로 사용, 수치 변경 금지
   - 범주형 x축의 경우 `fig.update_layout(xaxis_type="category")` 추가
   - 최종 변수명은 반드시 `fig`로 설정

4. **차트 설정 최소화:**
   - 기본 제목, 축 라벨만 설정
   - 복잡한 스타일링, 색상, 레이아웃 설정 지양
   - 간단하고 명확한 차트 생성에 집중

5. **에러 방지:**
   - 데이터 구조 확인: {{"rows": [...], "columns": [...]}} 형태인지 체크
   - 컬럼명 확인 후 사용: `df.columns.tolist()`로 실제 컬럼 확인
   - 데이터 타입 적절히 처리
   - 빈 데이터 예외 처리: `if df.empty: return`

6. **항상 시각화 시도:**
   - 어떤 데이터든 적절한 차트 생성
   - 불가능한 경우 테이블로 대체

코드만 반환하고 설명은 생략하세요.
''')

EXPLAINER_PROMPT = ('''
You are a data analysis assistant for a marketing team.

Your job is to explain the meaning of a data analysis result in clear, natural language.
Your explanation will be shown to a marketer who asked the original question.

Please use the following inputs:
- User's question: {question}
- instruction upon plotting graph: {instruction}
- Data values:
{data}
- plotly graph: 
{json_graph}

Based on the above, write a short, helpful answer to the User's question based on what the data shows.
Focus on **key trends**, **extreme values**, or **insightful comparisons**.

Answer in Korean.

Example:
Q: 가장 캠페인이 효과적이었던 요일이 언제였어?
A: 아래 차트는 요일별 전환율을 보여줍니다. 분석 결과, 토요일이 평균 전환율이 가장 높아 가장 효과적인 요일로 나타났습니다.

Now write the explanation:
''')

# ===== Nodes =====
def node_visualize(st: VisualizeState, llm) -> VisualizeState:
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 시각화 노드 시작 - 질문: %s", st.user_question)

    try:
        # JSON 데이터 검증
        data = json.loads(st.json_data)
        if not data:
            logger.warning("❌ 빈 데이터셋 - 시각화 불가능")
            st.error = "데이터가 비어있어 시각화할 수 없습니다."
            return st
        logger.info("✅ 데이터 검증 완료 - %d행 데이터", len(data))
    except json.JSONDecodeError as e:
        logger.error("❌ JSON 파싱 실패: %s", e)
        st.error = f"JSON 데이터 파싱 실패: {e}"
        return st

    prompt = VISUALIZER_PROMPT.format(
        question=st.user_question,
        json_data=st.json_data
    )

    logger.info("🤖 LLM 코드 생성 요청 시작")
    try:
        gen_code = llm.generate(prompt)
        if not gen_code or not str(gen_code).strip():
            logger.error("❌ LLM이 빈 응답 반환")
            st.error = "LLM이 시각화 코드를 반환하지 않았습니다."
            return st
        logger.info("✅ LLM 코드 생성 완료 - %d자", len(gen_code))
    except Exception as e:
        logger.error("❌ LLM 호출 실패: %s", e)
        st.error = f"LLM 호출 실패: {e}"
        return st

    # 코드 추출
    code = (
        gen_code.replace("```python", "")
                .replace("```", "")
                .strip()
    )
    logger.info("🔧 추출된 코드 길이: %d자", len(code))

    exec_env = {
        "pd": pd,
        "px": px,
        "go": go,
        "json": json,
        "data": data,
    }

    try:
        logger.info("🚀 코드 실행 시작")
        safe_exec(code, exec_env)    
        logger.info("✅ 코드 실행 완료")
    except RuntimeError as e:        
        logger.error("❌ 런타임 에러 - 테이블 폴백으로 전환: %s", e)
        try:
            df_fallback = pd.DataFrame(data)
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df_fallback.columns)),
                cells=dict(values=[df_fallback[col] for col in df_fallback.columns])
            )])
            exec_env["fig"] = fig
            logger.info("✅ 테이블 폴백 생성 완료")
        except Exception as fe:
            logger.error("❌ 테이블 폴백도 실패: %s", fe)
            st.error = f"시각화 및 테이블 생성 모두 실패: {e}, {fe}"
            return st
    except Exception as e:
        logger.error("❌ 코드 실행 중 예외 발생: %s", e)
        logger.error("실패한 코드:\n%s", code[:500])
        st.error = f"시각화 코드 실행 오류: {e}"
        return st

    # Figure 찾기
    fig = exec_env.get("fig")
    if fig is None:
        logger.warning("⚠️ 'fig' 변수 없음 - 환경에서 Figure 객체 검색")
        for k, v in exec_env.items():
            if isinstance(v, go.Figure):
                logger.info("✅ Figure 객체 발견: %s", k)
                fig = v
                break
    
    if fig is None:
        logger.error("❌ Plotly Figure 객체를 찾을 수 없음")
        logger.error("환경 변수: %s", list(exec_env.keys()))
        st.error = "Plotly Figure를 찾지 못했습니다."
        return st

    try:
        logger.info("🔄 Figure를 JSON으로 변환 중")
        
        # 매우 간소화된 레이아웃 설정
        fig.update_layout(
            template="none",  # 템플릿 제거
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False,  # 범례 제거 (단순한 차트용)
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        # 축 설정 단순화
        fig.update_xaxes(showgrid=False, showline=True, linecolor="black")
        fig.update_yaxes(showgrid=True, gridcolor="lightgray", showline=True, linecolor="black")
        
        # Plotly JSON 형식으로 생성 (이전 방식대로)
        st.json_graph = fig.to_json(remove_uids=True)
        
        # JSON 크기 로깅
        json_size_kb = len(st.json_graph) / 1024
        logger.info("✅ 간소화된 JSON 변환 완료 - %.1fKB", json_size_kb)
        
    except Exception as e:
        logger.error("❌ Figure JSON 변환 실패: %s", e)
        st.error = f"차트 JSON 변환 실패: {e}"
        return st
    
    st.error = ""
    logger.info("🎉 시각화 노드 성공적으로 완료")
    return st

def node_explain(st: VisualizeState, llm) -> VisualizeState:
    st.output = llm.generate(EXPLAINER_PROMPT.format(
        question=st.user_question,
        instruction=st.instruction,
        data=st.json_data,
        json_graph = st.json_graph
    ))
    return st

# ===== graph ======
def build_visualize_graph(model: str):
    llm = GeminiClient(model)
    g = StateGraph(VisualizeState)

    g.add_node("visualize", lambda s: node_visualize(s, llm))
    g.add_node("explain", lambda s: node_explain(s, llm))

    g.set_entry_point("visualize")
    g.add_edge('visualize', 'explain')
    g.add_edge('explain', END)

    return g.compile()