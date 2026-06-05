import os
from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Voice-Driven Revenue Cycle Copilot"
    VERSION: str = "1.0.0"
    
    # Render provides DATABASE_URL directly; local dev uses individual vars
    DATABASE_URL: str | None = None
    
    POSTGRES_USER: str = "admin"
    POSTGRES_PASSWORD: str = "password123"
    POSTGRES_SERVER: str = "localhost" # or 'db' for docker
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "revenue_cycle_db"
    
    # Free API Providers (Groq / Gemini)
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        # Render sets DATABASE_URL — use it if available
        if self.DATABASE_URL:
            url = self.DATABASE_URL
            # Render uses 'postgres://' but SQLAlchemy 2.x requires 'postgresql://'
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            return url
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

settings = Settings()
