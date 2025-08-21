from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import json 

from app.core.config import settings
from app.database.chat_history import save_chat_message

async def generate_chat_title(message: str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=settings.GOOGLE_API_KEY, max_retries=3)
        prompt = ChatPromptTemplate.from_template(
            "사용자의 첫 번째 메시지를 바탕으로, 대화의 주제를 잘 나타내는 간결한 한글 제목을 5단어 이내로 생성해줘. 제목만 따옴표 없이 반환해. 메시지: '{message}'"
        )
        chain = prompt | llm
        title_response = await chain.ainvoke({"message": message})
        title = title_response.content.strip().replace('"', '')
        return title if title else "새로운 대화"
    except Exception as e:
        print(f"Error generating chat title: {e}")
        return "새로운 대화"


async def stream_and_save_wrapper(chat_id: str, user_message: str, response_stream):
    """
    스트림을 클라이언트에 전달하면서, 실제 AI 응답 텍스트만 추출하여 DB에 저장합니다.
    """
    full_response_content = []
    async for chunk_str in response_stream:
        yield chunk_str 
        
        if chunk_str.startswith('data: '):
            try:
                data = json.loads(chunk_str[6:])
                if data.get('type') == 'chunk' and data.get('content'):
                    full_response_content.append(data['content'])
            except (json.JSONDecodeError, KeyError):
                continue 
    final_agent_message = "".join(full_response_content)

    if final_agent_message:
        save_chat_message(
            chat_id=chat_id, 
            user_message=user_message,
            agent_message=final_agent_message
        )
