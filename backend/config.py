import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self):
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.model_name: str = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        self.use_groq: bool = os.getenv("USE_GROQ", "True").lower() in ("true", "1", "yes")
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        self.log_level: str = os.getenv("LOG_LEVEL", "info")


settings = Settings()
