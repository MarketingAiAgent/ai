# 작동 방법 
## 환경 구성 
**터미널에서 아래 실행**
```Terminal 
conda create -n minti python=3.11
conda activate minti 
pip install -r requirements.txt
```
**IDE 설정에서 커널 일반 Python -> minti 로 변경** 

## 서버 시작
```
PYTHONPATH=app/ uvicorn main:app --reload
```

- `"[테스트용] 프로모션 계획"` - 프로모션 계획 테스트

### Mock 기능 제거
Mock 기능을 완전히 제거하려면:
1. `app/mock/` 폴더 삭제
2. `app/api/endpoints/chat.py`에서 mock 관련 import 제거
3. `app/core/config.py`에서 `ENABLE_MOCK_MODE` 설정 제거

## API 정의서 확인
**"접속주소"/docs**