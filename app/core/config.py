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
    
settings = Settings()

if PROFILE == "local":
    os.environ['TAVILY_API_KEY'] = settings.TAVILY_API_KEY