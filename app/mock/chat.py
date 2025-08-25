import json
import asyncio
from typing import AsyncGenerator

async def mock_suggestion() -> AsyncGenerator[str, None]:
    """[테스트용] 최종 확인에 대한 mock 응답"""
    message = '''
이것은 테스트용 csv url 입니다. [csv 다운로드] (https://rgmarketaiagentb767.blob.core.windows.net/minti-images/ad_daily.csv?sp=r&st=2025-08-25T06:38:34Z&se=2025-08-31T14:53:34Z&skoid=03d5b0e0-130c-4804-8b73-56ec3a3ff135&sktid=736d39e1-5c76-403f-9148-7432afb3f83b&skt=2025-08-25T06:38:34Z&ske=2025-08-31T14:53:34Z&sks=b&skv=2024-11-04&sv=2024-11-04&sr=b&sig=XyoY6G8HT1BErh1kA2Ya21nWUjqC4UHnJuEKpPirckM%3D)
최종 확인 테스트입니다. 프로모션 계획이 완성되었습니다. 다음 단계로 진행하시겠습니까?
'''

    
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
    payload = {"type": "done"}
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"