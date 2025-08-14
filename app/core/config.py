import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")
    
    PROJECT_NAME: str
    API_VERSION_STR: str
    LOG_LEVEL: str
    
    POSTGRES_USER: str 
    POSTGRES_PASSWORD: str 
    POSTGRES_SERVER: str 
    POSTGRES_PORT: int 
    POSTGRES_DB: str 

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str 
    ANTHROPIC_API_KEY: str 
    SUPABASE_API_KEY: str 

    # --- 테스트 용 --- 
    CONN_STR: str
    SCHEMA_INFO: str
    
settings = Settings()