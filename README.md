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

## API 정의서 확인
**"접속주소"/docs**