import re

from ai_companion.settings import settings
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser




def get_chat_model(temperature: float = 0.7):
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.TEXT_MODEL_NAME,
        temperature=temperature,
    )
    
    
    
def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    """Strip asterisk-wrapped content from LLM output."""

    def parse(self, text: str) -> str:
        return remove_asterisk_content(super().parse(text))
    
    