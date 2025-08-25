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
import logging

logger = logging.getLogger(__name__)

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
            max_output_tokens=max_output_tokens,
            # 일부 SDK 버전은 response_schema 미지원일 수 있음
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
아래는 사용자의 원래 질문, SQL 쿼리, 그리고 해당 쿼리 실행 결과의 JSON 데이터입니다.
이 정보를 바탕으로 파이썬의 **Plotly Express** 라이브러리를 사용하여 가장 적합한 시각화 코드를 생성해주세요.
⚠️ 중요한 규칙: 아래 JSON 데이터의 수치를 절대 변경하지 마세요. 있는 그대로 사용하세요.

**사용자의 원래 질문 (비즈니스 맥락):**
{question}

**쿼리 결과 (JSON 데이터 예시 - 상위 5개 행만):**
{json_data}

**시각화 코드 생성 요구사항:**
0.  print 함수 사용 금지
1.  데이터프레임을 로드하는 코드 (`df = pd.DataFrame(data)`)부터 시작해주세요. `data` 변수에는 전체 JSON 데이터가 리스트 형태로 들어있다고 가정합니다.
2.  데이터의 컬럼 이름, 데이터 타입, 그리고 사용자의 원래 질문을 고려하여 **가장 적절하다고 판단되는 차트 유형** (예: 막대 그래프, 선 그래프, 파이 차트, 산점도 등)을 선택해주세요.
3.  차트 유형에 맞게 **X축/Y축 또는 라벨/값에 가장 적절한 컬럼을 자동으로 선택**해주세요.
4.  차트 제목은 사용자의 질문과 데이터 내용을 반영하여 적절하게 생성해주세요.
5.  축 라벨도 컬럼의 의미에 맞게 적절하게 지정해주세요.
6.  x축이 월, 요일 등 범주형 데이터일 경우 `fig.update_layout(xaxis_type="category")`를 반드시 추가하여 자동 축 왜곡을 방지해주세요.
7.  모든 코드를 하나의 파이썬 코드 블록(` ```python `)으로 감싸서 반환해주세요.
8.  어떤 SQL 쿼리가 들어오든 **항상** 시각화 코드를 생성하려고 시도해주세요. 만약 시각화에 적합하지 않은 데이터라고 판단되면, 왜 그런지 간략히 주석으로 설명하고 간단한 테이블 출력 코드라도 생성해주세요.
''')

# ===== Nodes =====
def node_visualize(st: VisualizeState, llm) -> VisualizeState:

    prompt = VISUALIZER_PROMPT.format(
        question = st.user_question,
        json_data = st.json_data,
        name = uuid4().hex
    )

    gen_code = llm.generate(prompt)
    if not gen_code or not str(gen_code).strip():
        st.error = "LLM이 시각화 코드를 반환하지 않았습니다."
        logger.error(f"LLM이 시각화 코드를 반환하지 않았습니다. {gen_code}")
        return st

    code = (
        gen_code.replace("```python", "")
                .replace("```", "")
                .strip()
    )

    exec_env = {
        "pd": pd,
        "px": px,
        "go": go,
        "json": json,
        "data": json.loads(st.json_data),
    }

    try:
        safe_exec(code, exec_env)    
    except RuntimeError as e:        
        df_fallback = pd.DataFrame(json.loads(st.json_data))
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df_fallback.columns)),
            cells=dict(values=[df_fallback[col] for col in df_fallback.columns])
        )])
        exec_env["fig"] = fig       
    except Exception as e:
        st.error = f"시각화 코드 실행 오류: {e}"
        return st

    fig = exec_env.get("fig")
    if fig is None:
        for v in exec_env.values():
            if isinstance(v, go.Figure):
                fig = v
                break
    if fig is None:
        st.error = "Plotly Figure 를 찾지 못했습니다."
        return st

    st.json_graph = fig.to_json()
    st.error = ""
    return st

# ===== graph ======
def build_visualize_graph(model: str):
    llm = GeminiClient(model)
    g = StateGraph(VisualizeState)

    g.add_node("visualize", lambda s: node_visualize(s, llm))

    g.set_entry_point("visualize")
    g.add_edge('visualize', END)

    return g.compile()