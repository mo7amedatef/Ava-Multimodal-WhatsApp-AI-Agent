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
    
    
    ITT_MAX_TOKENS: int = 1000
    STT_LANGUAGE: str = "en"
    
    
    # WhatsApp Cloud API settings
    WHATSAPP_TOKEN: str
    WHATSAPP_PHONE_NUMBER_ID: str
    WHATSAPP_VERIFY_TOKEN: str
    WHATSAPP_API_VERSION: str = "v21.0"
    WHATSAPP_GRAPH_API_URL: str = "https://graph.facebook.com"
    WHATSAPP_HTTP_TIMEOUT_SECONDS: float = 30.0
    WHATSAPP_MAX_TEXT_LENGTH: int = 4096
    WHATSAPP_MAX_CAPTION_LENGTH: int = 1024
    WHATSAPP_AUDIO_MIME_TYPE: str = "audio/mpeg"
    WHATSAPP_IMAGE_MIME_TYPE: str = "image/png"
    WHATSAPP_IMAGE_ANALYSIS_PROMPT: str = (
        "Please describe what you see in this image in the context of our conversation."
    )
    
    
    WHATSAPP_DEDUP_CACHE_SIZE: int = 1000
    WHATSAPP_AUDIO_FILENAME: str = "response.mp3"
    WHATSAPP_IMAGE_FILENAME: str = "response.png"
    WHATSAPP_UNSUPPORTED_MESSAGE_REPLY: str = (
        "Sorry, I can only understand text, voice notes and images for now."
    )
    WHATSAPP_PROCESSING_ERROR_REPLY: str = "Sorry, something went wrong on my side. Please try again in a moment."

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"

settings = Settings()