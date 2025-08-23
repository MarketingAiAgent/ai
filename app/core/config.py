import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
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