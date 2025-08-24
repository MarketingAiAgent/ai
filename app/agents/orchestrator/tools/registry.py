
"""
도구 레지스트리 (v2).
실제 실행은 기존 orchestrator.tools 모듈의 함수를 그대로 사용합니다.
"""
from typing import Callable, Dict, Any
from app.agents.orchestrator.state import OrchestratorState

# 원본 도구 구현
from app.agents.orchestrator.tools import (  # type: ignore
    run_t2s_agent_with_instruction,
    run_tavily_search,
    scrape_webpages,
    marketing_trend_search,
    beauty_youtuber_trend_search,
)

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]

def build_tool_map(state: OrchestratorState) -> Dict[str, ToolFn]:
    """현재 상태를 캡처하는 람다로 래핑하여 기존 시그니처와 호환되게 만듭니다."""
    return {
        "t2s": lambda args: run_t2s_agent_with_instruction(state, args.get("instruction", "")),
        "tavily_search": lambda args: run_tavily_search(args.get("query", ""), args.get("max_results", 5)),
        "scrape_webpages": lambda args: scrape_webpages(args.get("urls", [])),
        "marketing_trend_search": lambda args: marketing_trend_search(args.get("question", "")),
        "beauty_youtuber_trend_search": lambda args: beauty_youtuber_trend_search(args.get("question", "")),
    }
