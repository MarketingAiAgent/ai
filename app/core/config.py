import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")
    
    PROJECT_NAME: str
    API_VERSION_STR: str
    LOG_LEVEL: str
    
    COSMOS_DB_CONNECTION_STRING:str

    FORMATTER_SUPERBASE_URL:str
    FORMATTER_SUPABASE_ANON_KEY:str
    
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str 
    ANTHROPIC_API_KEY: str 
    SUPABASE_API_KEY: str 

    # --- 테스트 용 --- 
    CONN_STR: str
    SCHEMA_INFO: str
    
settings = Settings()