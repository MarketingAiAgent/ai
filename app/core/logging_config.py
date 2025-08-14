# app/core/logging_config.py

import logging.config
from .config import settings 

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
        }
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
        }
    },
    "loggers": {
        "uvicorn": {"level": settings.LOG_LEVEL, "handlers": ["default"], "propagate": False},
        "uvicorn.error": {"level": settings.LOG_LEVEL, "handlers": ["default"], "propagate": False},
        "uvicorn.access": {"level": settings.LOG_LEVEL, "handlers": ["access"], "propagate": False}
    },
}

def setup_logging():
    """애플리케이션에 로깅 설정을 적용하는 함수"""
    logging.config.dictConfig(LOGGING_CONFIG)