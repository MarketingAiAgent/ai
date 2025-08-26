import os
from pydantic_settings import BaseSettings, SettingsConfigDict

PROFILE = os.getenv("PROFILE", "local")

class Settings(BaseSettings):
    
    if PROFILE == "local":
        model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")
        
    PROJECT_NAME: str
    API_VERSION_STR: str
    LOG_LEVEL: str
    
    COSMOS_DB_CONNECTION_STRING: str

    SUPERBASE_URL: str
    SUPABASE_ANON_KEY: str
    
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str 
    ANTHROPIC_API_KEY: str 
    TAVILY_API_KEY: str

    CONN_STR: str
    SCHEMA_INFO: str
    
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_NAME: str = "exports"
    
    # Mock 모드 설정 (기본값: False)
    ENABLE_MOCK_MODE: bool = True
    
settings = Settings()

if PROFILE == "local":
    os.environ['TAVILY_API_KEY'] = settings.TAVILY_API_KEY