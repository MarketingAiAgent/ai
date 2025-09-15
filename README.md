# Minti - FastAPI 서버

## 📌 개요
마케터를 위한 AI 에이전트 프로젝트 "Minti"의 AI 서버입니다. 
AI 기능들이 구현되었습니다.

## 🛠 기술 스택
- **Backend**: Python 3.11, FastAPI
- **Agent**: LangGraph, CrewAI
- **Database**: PostgreSQL, Supabase, MongoDB (CosmosDB)
- **Data Visualization**: Matplotlib, Seaborn, Plotly, Pandas
- **Infrastructure**: Docker, Kubernetes, Azure Storage
- **External APIs**: Tavily Search, BeautifulSoup4
- **DevOps**: Github Actions
  
## ✨ 주요 기능
- **자연어 기반 데이터 조회**: SQL 자동 생성을 통한 마케팅 데이터 분석
- **실시간 스트리밍 채팅**: FastAPI 기반 실시간 대화형 인터페이스
- **AI 프로모션 기획**: 브랜드/카테고리별 맞춤형 마케팅 캠페인 자동 생성
- **트렌드 기반 분석**: 웹 검색 및 지식 DB를 활용한 최신 마케팅 트렌드 반영
- **데이터 시각화**: 차트, 그래프 자동 생성 및 파일 내보내기
- **멀티 에이전트 오케스트레이션**: LangGraph 기반 복합 작업 처리

## 📂 프로젝트 구조
```
app/
├── agents/                    # AI 에이전트 모듈
│   ├── orchestrator/         # 메인 오케스트레이터 에이전트
│   │   ├── graph.py          # LangGraph 워크플로우 정의
│   │   ├── state.py          # 상태 관리
│   │   ├── tools.py          # 외부 도구 연동
│   │   └── helpers.py        # 헬퍼 함수들
│   ├── text_to_sql/          # 자연어-SQL 변환 에이전트
│   │   ├── crew.py           # CrewAI 기반 SQL 생성
│   │   └── graph.py          # SQL 실행 그래프
│   ├── promotion/            # 프로모션 기획 에이전트
│   │   └── state.py          # 프로모션 상태 관리
│   ├── formatter/            # 응답 포맷팅 에이전트
│   │   ├── grapy.py          # 그래프 생성
│   │   └── state.py          # 포맷터 상태
│   └── visualizer/           # 데이터 시각화 에이전트
│       ├── graph.py          # 시각화 워크플로우
│       └── state.py          # 시각화 상태
├── api/                      # REST API 엔드포인트
│   └── endpoints/
│       ├── chat.py           # 채팅 관련 API
│       └── design.py         # 디자인 관련 API
├── core/                     # 핵심 설정
│   ├── config.py             # 환경 설정
│   └── logging_config.py     # 로깅 설정
├── database/                 # 데이터베이스 연동
│   ├── connection.py         # DB 연결 관리
│   ├── chat_history.py       # 채팅 이력 관리
│   ├── plans.py              # 프로모션 계획 저장
│   └── supabase.py           # Supabase 연동
├── schema/                   # 데이터 스키마
│   ├── _base.py              # 기본 스키마
│   └── chat.py               # 채팅 스키마
├── service/                  # 비즈니스 로직
│   └── chat_service.py       # 채팅 서비스
├── utils/                    # 유틸리티
│   └── blob_storage.py       # 파일 저장소 관리
├── mock/                     # 테스트용 Mock 데이터
└── main.py                   # FastAPI 애플리케이션 진입점
k8s/                          # Kubernetes 배포 설정
├── configmap.yml             # 환경 변수 설정
├── deployment.yml            # 애플리케이션 배포
├── service.yml               # 서비스 노출
└── network-policy.yml        # 네트워크 정책
```
