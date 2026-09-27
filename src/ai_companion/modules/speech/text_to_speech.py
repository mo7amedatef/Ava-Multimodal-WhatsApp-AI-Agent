import asyncio
from typing import Optional

from ai_companion.core.exceptions import TextToSpeechError
from ai_companion.settings import settings
from elevenlabs import ElevenLabs, VoiceSettings


class TextToSpeech:
    """A class to handle text-to-speech conversion using ElevenLabs."""

    def __init__(self):
        """Initialize the TextToSpeech class."""
        self._client: Optional[ElevenLabs] = None

    @property
    def client(self) -> ElevenLabs:
        """Get or create ElevenLabs client instance using singleton pattern."""
        if self._client is None:
            self._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return self._client

    def _convert(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs synchronously."""
        return b"".join(
            self.client.text_to_speech.convert(
                voice_id=settings.ELEVENLABS_VOICE_ID,
                text=text,
                model_id=settings.TTS_MODEL_NAME,
                voice_settings=VoiceSettings(
                    stability=settings.TTS_STABILITY,
                    similarity_boost=settings.TTS_SIMILARITY_BOOST,
                ),
            )
        )

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio data

        Raises:
            ValueError: If the input text is empty or too long
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > settings.TTS_MAX_CHARACTERS:
            raise ValueError(f"Input text exceeds maximum length of {settings.TTS_MAX_CHARACTERS} characters")

        try:
            audio_bytes = await asyncio.to_thread(self._convert, text)

            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            return audio_bytes

        except Exception as e:
            raise TextToSpeechError(f"Text-to-speech conversion failed: {str(e)}") from e