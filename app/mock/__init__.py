from typing import Optional, AsyncGenerator
from app.core.config import settings
from .chat import  mock_suggestion

def get_mock_response(message: str) -> Optional[AsyncGenerator[str, None]]:
    """
    테스트 메시지면 mock 응답 반환, 아니면 None
    
    Args:
        message: 사용자 메시지
        
    Returns:
        Mock 응답 스트림 또는 None
    """
    if not settings.ENABLE_MOCK_MODE:
        return None
    
    # 테스트 패턴과 해당하는 mock 함수 매핑
    patterns = {
        "[테스트용] 최종 확인": mock_suggestion,
    }
    
    # 패턴 매칭
    for pattern, mock_func in patterns.items():
        if pattern in message:
            return mock_func()
    
    return None

async def mock_stream_with_save(chat_id: str, user_message: str, mock_stream: AsyncGenerator[str, None]):
    """
    Mock 응답을 스트리밍하면서 채팅 히스토리에 저장
    
    Args:
        chat_id: 채팅 ID
        user_message: 사용자 메시지
        mock_stream: Mock 응답 스트림
    """
    from app.database.chat_history import save_chat_message
    import json
    
    full_response_content = []
    
    async for chunk_str in mock_stream:
        yield chunk_str
        
        # 응답 내용 수집
        if chunk_str.startswith('data: '):
            try:
                data = json.loads(chunk_str[6:])
                if data.get('type') == 'chunk' and data.get('content'):
                    full_response_content.append(data['content'])
            except (json.JSONDecodeError, KeyError):
                continue
    
    # 채팅 히스토리에 저장
    final_agent_message = "".join(full_response_content)
    if final_agent_message:
        save_chat_message(
            chat_id=chat_id,
            user_message=user_message,
            agent_message=final_agent_message
        )
