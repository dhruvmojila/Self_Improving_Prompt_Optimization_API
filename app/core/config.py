"""
Core configuration settings for the Prompt Optimization API.
Loads settings from environment variables with defaults.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import pathlib


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Supports Groq and Gemini LLM providers with rate limiting.
    """
    
    # Application
    APP_NAME: str = "Prompt Optimization API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # LLM Provider Configuration
    GROQ_API_KEY: str
    GEMINI_API_KEY: str
    
    # Teacher Models (more capable, used for optimization)
    TEACHER_MODELS: List[str] = [
        "llama-3.3-70b-versatile",  # Groq
        "openai/gpt-oss-120b",       # Groq  
        "gemini-2.0-flash-exp"            # Gemini
    ]
    DEFAULT_TEACHER_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_TEACHER_PROVIDER: str = "groq"  # "groq" or "gemini"
    
    # Student Models (efficient, used for inference)
    STUDENT_MODELS: List[str] = [
        "llama-3.1-8b-instant",        # Groq
        "gemini-2.0-flash-exp"  # Gemini
    ]
    DEFAULT_STUDENT_MODEL: str = "llama-3.1-8b-instant"
    DEFAULT_STUDENT_PROVIDER: str = "groq"
    
    # Judge Models (for LLM-as-a-Judge evaluation)
    JUDGE_MODELS: List[str] = [
        "llama-3.3-70b-versatile",  # Groq
        "gemini-2.0-flash-exp"            # Gemini
    ]
    DEFAULT_JUDGE_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_JUDGE_PROVIDER: str = "groq"
    
    # Rate Limiting (requests per minute)
    GROQ_RPM_LIMIT: int = 30  # Groq free tier
    GROQ_TOKENS_PER_MINUTE: int = 20000
    GEMINI_RPM_LIMIT: int = 15  # Gemini free tier
    GEMINI_TOKENS_PER_MINUTE: int = 1000000
    
    # Retry Configuration
    MAX_RETRIES: int = 5
    RETRY_MIN_WAIT_SECONDS: int = 1
    RETRY_MAX_WAIT_SECONDS: int = 60
    
    # Pixeltable Configuration
    PIXELTABLE_DB_PATH: str = "./pixeltable_db/pgdata"
    PIXELTABLE_HOME: Optional[str] = "./pixeltable_db"

    # Redis/Celery Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    
    # Storage Configuration
    ARTIFACTS_DIR: pathlib.Path = Field(default_factory=lambda: pathlib.Path("./artifacts"))
    PROMPT_ARTIFACTS_DIR: pathlib.Path = Field(default_factory=lambda: pathlib.Path("./artifacts/prompts"))
    DATASET_ARTIFACTS_DIR: pathlib.Path = Field(default_factory=lambda: pathlib.Path("./artifacts/datasets"))
    
    # LangSmith (optional for tracing)
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "prompt-optimization"
    LANGSMITH_TRACING: bool = False
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Optimization Defaults
    DEFAULT_OPTIMIZER_TYPE: str = "bootstrap"  # "bootstrap" or "mipro"
    DEFAULT_TRAIN_SPLIT: float = 0.6
    DEFAULT_DEV_SPLIT: float = 0.2
    DEFAULT_TEST_SPLIT: float = 0.2
    MAX_FEW_SHOT_EXAMPLES: int = 5
    
    # Budget Controls
    DEFAULT_BUDGET_USD: float = 5.0
    MAX_BUDGET_USD: float = 100.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Auto-construct Celery URLs if not provided
        if not self.CELERY_BROKER_URL:
            self.CELERY_BROKER_URL = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        
        if not self.CELERY_RESULT_BACKEND:
            self.CELERY_RESULT_BACKEND = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB + 1}"
        
        # Create artifacts directories
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.PROMPT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATASET_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
