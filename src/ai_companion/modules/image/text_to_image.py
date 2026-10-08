import asyncio
import base64
import logging
import os
from typing import Optional

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from ai_companion.settings import settings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from together import Together


class ScenarioPrompt(BaseModel):
    """Class for the scenario response."""

    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")


class EnhancedPrompt(BaseModel):
    """Class for the enhanced text prompt."""

    content: str = Field(..., description="The enhanced text prompt to generate an image")


class TextToImage:
    """A class to handle text-to-image generation using Together AI."""

    def __init__(self):
        """Initialize the TextToImage class."""
        self._together_client: Optional[Together] = None
        self.logger = logging.getLogger(__name__)

    @property
    def together_client(self) -> Together:
        """Get or create Together client instance using singleton pattern."""
        if self._together_client is None:
            self._together_client = Together(api_key=settings.TOGETHER_API_KEY)
        return self._together_client

    def _get_llm(self, temperature: float) -> ChatGroq:
        """Create a ChatGroq instance with the given temperature."""
        return ChatGroq(
            model=settings.TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=temperature,
            max_retries=2,
        )

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image from a prompt using Together AI."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            self.logger.info(f"Generating image for prompt: '{prompt}'")

            response = await asyncio.to_thread(
                self.together_client.images.generate,
                prompt=prompt,
                model=settings.TTI_MODEL_NAME,
                width=settings.TTI_IMAGE_WIDTH,
                height=settings.TTI_IMAGE_HEIGHT,
                steps=settings.TTI_STEPS,
                n=1,
                response_format="b64_json",
            )

            image_data = base64.b64decode(response.data[0].b64_json)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(image_data)
                self.logger.info(f"Image saved to {output_path}")

            return image_data

        except Exception as e:
            raise TextToImageError(f"Failed to generate image: {str(e)}") from e

    async def create_scenario(self, chat_history: Optional[list] = None) -> ScenarioPrompt:
        """Create a first-person narrative scenario and image prompt based on chat history."""
        try:
            def _sanitize_content(text: str) -> str:
                """Strip newlines and escape role-like prefixes to prevent
                users from injecting fake System/Human/AI messages."""
                sanitized = text.replace("\n", " ").replace("\r", " ")
                # Escape any attempt to inject role prefixes
                for prefix in ("System:", "Human:", "Ai:", "Assistant:", "User:"):
                    sanitized = sanitized.replace(prefix, prefix.replace(":", "∶"))
                return sanitized.strip()

            formatted_history = "\n".join(
                [f"[{msg.type.upper()}] {_sanitize_content(msg.content)}" for msg in chat_history]
            )
            self.logger.info("Creating scenario from chat history")

            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT,
                )
                | self._get_llm(temperature=0.4).with_structured_output(ScenarioPrompt)
            )

            scenario = await chain.ainvoke({"chat_history": formatted_history})
            self.logger.info(f"Created scenario: {scenario}")

            return scenario

        except Exception as e:
            raise TextToImageError(f"Failed to create scenario: {str(e)}") from e

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        try:
            self.logger.info(f"Enhancing prompt: '{prompt}'")

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | self._get_llm(temperature=0.25).with_structured_output(EnhancedPrompt)
            )

            enhanced_prompt = (await chain.ainvoke({"prompt": prompt})).content
            self.logger.info(f"Enhanced prompt: '{enhanced_prompt}'")

            return enhanced_prompt

        except Exception as e:
            raise TextToImageError(f"Failed to enhance prompt: {str(e)}") from e