from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import asyncio

from app.core.config import settings 
from app.core.logging_config import setup_logging
from app.api.endpoints import chat 

from typing import AsyncGenerator

setup_logging() 

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # HTML, CSS, JavaScript 코드를 파이썬 문자열로 정의
    # app/main.py 의 html_content 변수

    # app/main.py 의 html_content 변수

    # app/main.py에 들어갈 html_content 변수

    # app/main.py에 들어갈 html_content 변수

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background-color: #f4f4f9; display: flex; justify-content: center; align-items: center; height: 100vh; }
        #chat-container { width: 100%; max-width: 700px; height: 90vh; background-color: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        #messages { flex: 1; overflow-y: auto; padding: 20px; border-bottom: 1px solid #ddd; }
        .message { margin-bottom: 15px; display: flex; flex-direction: column; }
        .user-message { align-items: flex-end; }
        .user-message p { margin: 0; padding: 10px 15px; border-radius: 18px; max-width: 80%; line-height: 1.5; background-color: #007bff; color: white; }
        .ai-message { align-items: flex-start; }
        .ai-message-container { background-color: #e9e9eb; color: #333; padding: 1px; border-radius: 18px; max-width: 100%; }
        .ai-message-container h3 { margin-top: 15px; padding: 0 15px;}
        .ai-message-container ul { padding-left: 35px; }
        .ai-message-container p { padding: 0 15px; }
        #chat-form { display: flex; padding: 20px; border-top: 1px solid #ddd; }
        #message-input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 20px; margin-right: 10px; font-size: 16px; }
        #chat-form button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 20px; cursor: pointer; font-size: 16px; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="messages">
            <div class="message ai-message">
                <div class="ai-message-container">
                    <p style="padding-top: 10px;">안녕하세요! 마케팅 분석 AI 에이전트입니다.</p>
                </div>
            </div>
        </div>
        <form id="chat-form">
            <input type="text" id="message-input" placeholder="메시지를 입력하세요..." autocomplete="off">
            <button type="submit">전송</button>
        </form>
    </div>

    <script>
        const thread_id = crypto.randomUUID();
        const form = document.getElementById('chat-form');
        const input = document.getElementById('message-input');
        const messages = document.getElementById('messages');

        function addUserMessage(text) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', 'user-message');
            const p = document.createElement('p');
            p.textContent = text;
            messageDiv.appendChild(p);
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }

        form.addEventListener('submit', async function(event) {
            event.preventDefault();
            const userMessage = input.value.trim();
            if (!userMessage) return;

            addUserMessage(userMessage);
            input.value = '';

            // AI 응답을 담을 변수들을 미리 선언
            let aiReportContainer = null;
            let aiTextElement = null;
            let accumulatedText = '';

            try {
                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        "user_message": userMessage,
                        "thread_id": thread_id,
                        "org_id": "test-org",
                        "db_connection_string": "placeholder"
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, {stream: true});
                    
                    while (buffer.includes('\\n\\n')) {
                        const messageEndIndex = buffer.indexOf('\\n\\n');
                        const rawMessage = buffer.substring(0, messageEndIndex);
                        buffer = buffer.substring(messageEndIndex + 2);

                        if (rawMessage.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(rawMessage.substring(6));

                                if (data.type === 'report_start') {
                                    const messageDiv = document.createElement('div');
                                    messageDiv.classList.add('message', 'ai-message');
                                    aiReportContainer = document.createElement('div');
                                    aiReportContainer.classList.add('ai-message-container');
                                    messageDiv.appendChild(aiReportContainer);
                                    messages.appendChild(messageDiv);

                                } else if (data.type === 'text_chunk') {
                                    if (!aiTextElement) {
                                        aiTextElement = document.createElement('div');
                                        aiReportContainer.appendChild(aiTextElement);
                                    }
                                    accumulatedText += data.content;
                                    aiTextElement.innerHTML = marked.parse(accumulatedText);

                                } else if (data.type === 'graph') {
                                    const graphDiv = document.createElement('div');
                                    graphDiv.id = 'graph-' + crypto.randomUUID();
                                    aiReportContainer.appendChild(graphDiv);
                                    Plotly.newPlot(graphDiv.id, data.content.data, data.content.layout, {responsive: true});
                                
                                } else if (data.type === 'done') {
                                    return;
                                }
                                messages.scrollTop = messages.scrollHeight;
                            } catch (e) {
                                console.error("JSON Parse Error:", e, "Data:", rawMessage);
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Fetch Error:', error);
                addUserMessage('오류가 발생했습니다: ' + error.message, 'ai');
            }
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)