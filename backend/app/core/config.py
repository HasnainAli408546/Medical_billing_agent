import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Voice-Driven Revenue Cycle Copilot"
    VERSION: str = "1.0.0"
    
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "password123"
    POSTGRES_SERVER: str = "localhost" # or 'db' for docker
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "revenue_cycle_db"
    
    # Free API Providers (Groq / Gemini)
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
