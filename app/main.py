from fastapi import FastAPI, Request, status
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import asyncio
import logging 
from app.core.config import settings 
from app.core.logging_config import setup_logging
from app.api.endpoints import chat 

from typing import AsyncGenerator

setup_logging() 

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME, 
)

app.include_router(chat.router)

async def word_stream(text: str) -> AsyncGenerator[str, None]:
    for w in text.split(): 
        yield f"data: {w}\n\n".encode("utf-8")
        await asyncio.sleep(0.02)
    yield "data: [DONE]\n\n"

@app.get("/healthz")
async def healthz():
    return {"ok": True}