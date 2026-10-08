import logging
import uuid
from datetime import datetime
from typing import List, Optional
from functools import lru_cache
from pydantic import BaseModel, Field

from ai_companion.core.prompts import MEMORY_ANALYSIS_PROMPT
from ai_companion.modules.memory.long_term.vector_store import get_vector_store
from ai_companion.settings import settings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""

    is_important: bool = Field(
        ...,
        description="Whether the message is important enough to be stored as a memory",
    )
    formatted_memory: Optional[str] = Field(..., description="The formatted memory to be stored")


class MemoryManager:
    """Manager class for handling long-term memory operations."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed.

        Uses structured chat messages (system + human) instead of raw string
        formatting to prevent prompt injection attacks.
        """
        messages = [
            SystemMessage(content=MEMORY_ANALYSIS_PROMPT),
            HumanMessage(content=message),
        ]
        return await self.llm.ainvoke(messages)

    async def extract_and_store_memories(self, message: BaseMessage, user_id: str) -> None:
        """Extract important information from a message and store in vector store.

        Args:
            message: The message to analyze
            user_id: The user who owns this memory
        """
        if message.type != "human":
            return

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(message.content)
        if analysis.is_important and analysis.formatted_memory:
            # Check if similar memory exists for this user
            similar = self.vector_store.find_similar_memory(analysis.formatted_memory, user_id=user_id)
            if similar:
                # Skip storage if we already have a similar memory
                self.logger.info(f"Similar memory already exists for user '{user_id}': '{analysis.formatted_memory}'")
                return

            # Store new memory scoped to this user
            self.logger.info(f"Storing new memory for user '{user_id}': '{analysis.formatted_memory}'")
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                },
                user_id=user_id,
            )

    def get_relevant_memories(self, context: str, user_id: str) -> List[str]:
        """Retrieve relevant memories for a specific user based on the current context.

        Args:
            context: The conversation context to search against
            user_id: The user whose memories to search
        """
        memories = self.vector_store.search_memories(context, user_id=user_id, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)

@lru_cache
def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance."""
    return MemoryManager()

