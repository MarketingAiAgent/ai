import json
import asyncio
from typing import AsyncGenerator

async def mock_suggestion() -> AsyncGenerator[str, None]:

  """[테스트용] 최종 확인에 대한 mock 응답"""
  message = '''
이것은 테스트용 csv url 입니다. [csv 다운로드](https://rgmarketaiagentb767.blob.core.windows.net/minti-images/ad_daily.csv?sp=r&st=2025-08-25T06:38:34Z&se=2025-08-31T14:53:34Z&skoid=03d5b0e0-130c-4804-8b73-56ec3a3ff135&sktid=736d39e1-5c76-403f-9148-7432afb3f83b&skt=2025-08-25T06:38:34Z&ske=2025-08-31T14:53:34Z&sks=b&skv=2024-11-04&sv=2024-11-04&sr=b&sig=XyoY6G8HT1BErh1kA2Ya21nWUjqC4UHnJuEKpPirckM%3D)
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

  payload = {"type": "table_start"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

  Table = '''
| product_id | product_name | brand | category_l1 | category_l2 | total_purchase_count |
|---|---|---|---|---|---|
| 26193 | 트리거포인트 MBX 마사지볼 | 트리거포인트 | 헬스,건강용품 | 마사지,보호대 | 15.0 |
| 5234 | 조르단 어린이 버디1 1P (색상랜덤) | 조르단 | 구강용품 | 칫솔 | 14.0 |
| 8052 | 프레쉬라이트 폼 쿨블루 (염색) | 프레쉬라이트 | 헤어케어 | 염색약,펌 | 14.0 |
| 11952 | 디어러스 쉬어 네일 10종 | 디어러스 | 네일 | 일반네일 | 14.0 |
| 26680 | 아베다 스칼프솔루션 리프레싱 프로텍티브 미스트 100ml | 아베다 | 헤어케어 | 트리트먼트,팩 | 14.0 |
'''

  for char in Table:
      payload = {
          "type": "chunk",
          "content": char
      }
      yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
      await asyncio.sleep(0.02)  # 타이핑 효과

  payload = {"type": "table_end"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

  simple_graph = {
    "data": [
      {
        "x": ["상품A", "상품B", "상품C", "상품D", "상품E"],
        "y": [15, 14, 12, 10, 8],
        "type": "bar",
        "marker": {
          "color": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A"]
        }
      }
    ],
    "layout": {
      "title": {
        "text": "상품별 구매 횟수",
        "font": {"size": 16}
      },
      "xaxis": {
        "title": "상품명",
        "tickangle": -45
      },
      "yaxis": {
        "title": "구매 횟수"
      },
      "margin": {
        "l": 50,
        "r": 50,
        "t": 80,
        "b": 80
      },
      "plot_bgcolor": "white",
      "paper_bgcolor": "white"
    }
  }
  
  payload = {'type': "graph", "content": simple_graph}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
  
  payload = {"type": "plan", "content": "brand"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # 완료 신호
  payload = {"type": "done"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"