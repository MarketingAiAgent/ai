
# orchestrator_v2 (비침투 리팩터링)

- **목표**: 내부 로직은 변경하지 않고, 그래프 조립/라우터/노드/툴/헬퍼 구조를 분리합니다.
- **방법**: 기존 `app.agents.orchestrator`의 함수들을 thin wrapper로 감싸서 호출합니다.

## 사용 방법 (권장: 점진적 전환)
1) 기존 코드는 그대로 둡니다.
2) orchestrator 앱을 사용할 곳에서 아래와 같이 import 경로만 바꾸세요.
   ```python
   from app.agents.orchestrator_v2.graph import orchestrator_app
   ```
3) 문제 없이 동작하는지 확인한 뒤, 프롬프트/노드 구현을 점차 `orchestrator_v2` 내부로 옮기고
   기존 모듈 의존을 줄여나가세요.

## 주의
- 이 버전은 **로직을 변경하지 않습니다.** 모든 노드/라우터는 기존 구현을 그대로 호출합니다.
- 장기적으로는 기존 `graph.py`의 라우터/노드 구현을 `orchestrator_v2/nodes/*.py`로 옮기고,
  테스트 통과 후 원본 의존을 제거하세요.
