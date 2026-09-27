from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_file_encoding="utf-8"
        )

    # API Keys
    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str
    TOGETHER_API_KEY: str

    # Qdrant settings
    QDRANT_API_KEY: str | None
    QDRANT_URL: str
    QDRANT_PORT: int = 6333
    QDRANT_HOST: str | None = None

    # Model settings
    TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"
    SMALL_TEXT_MODEL_NAME: str = "gemma2-9b-it"
    STT_MODEL_NAME: str = "whisper-large-v3-turbo"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"
    ITT_MODEL_NAME: str = "llama-3.2-90b-vision-preview"

    # Memory settings
    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5
    
    
    TTI_IMAGE_WIDTH: int = 1024
    TTI_IMAGE_HEIGHT: int = 768
    TTI_STEPS: int = 4
    OUTPUT_DIR: str = "generated_images"
    
    
    TTS_MAX_CHARACTERS: int = 5000
    TTS_STABILITY: float = 0.5
    TTS_SIMILARITY_BOOST: float = 0.5

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"

settings = Settings()