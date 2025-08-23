import json
import asyncio
from typing import AsyncGenerator

async def mock_suggestion() -> AsyncGenerator[str, None]:
    """[테스트용] 최종 확인에 대한 mock 응답"""
    message = "최종 확인 테스트입니다. 프로모션 계획이 완성되었습니다. 다음 단계로 진행하시겠습니까?"
    
    payload = {"type": "start"}
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    
    # 스트리밍 형태로 응답
    for char in message:
        payload = {
            "type": "chunk",
            "content": char
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.02)  # 타이핑 효과
    
    payload = {"type": "plan", "content": "brand"}
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    # 완료 신호
    yield "data: [DONE]\n\n"
