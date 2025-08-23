# app/core/logging_config.py
import logging.config
from .config import settings 

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # 타 로거 살려둡니다
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        # Uvicorn 로거들
        "uvicorn":        {"level": settings.LOG_LEVEL, "handlers": ["default"], "propagate": False},
        "uvicorn.error":  {"level": settings.LOG_LEVEL, "handlers": ["default"], "propagate": False},
        "uvicorn.access": {"level": settings.LOG_LEVEL, "handlers": ["access"],  "propagate": False},

        # (선택) 시끄러운 외부 로거 소거
        "httpx":   {"level": "WARNING", "handlers": ["default"], "propagate": False},
        "langchain": {"level": "WARNING", "handlers": ["default"], "propagate": False},
        "pymongo": {"level": "WARNING", "handlers": ["default"], "propagate": False},

        # ✅ 애플리케이션 네임스페이스(예: app.*) 직접 핸들러 연결
        "app": {"level": settings.LOG_LEVEL, "handlers": ["default"], "propagate": False},
    },
    # ✅ 루트 로거: app.* 이외의 모듈로거도 여기로 올라오면 찍힙니다.
    "root": {"level": settings.LOG_LEVEL, "handlers": ["default"]},
}

def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
