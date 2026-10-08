from typing import Literal, Optional

from langgraph.graph import MessagesState


class AICompanionState(MessagesState):
    """State class for the AI Companion workflow.

    Extends MessagesState to track conversation history.

    Attributes:
        summary (str): Running summary of the conversation.
        workflow (str): Active workflow — 'conversation', 'image', or 'audio'.
        audio_buffer (Optional[bytes]): Raw audio data for STT, present only on voice inputs.
        image_path (Optional[str]): Path to the image file, present only on image interactions.
        current_activity (str): Ava's current activity based on her schedule.
        apply_activity (bool): Whether to inject the current activity into the prompt.
        memory_context (str): Retrieved long-term memories to inject into the character card.
    """

    summary: str
    workflow: Literal["conversation", "image", "audio"]
    audio_buffer: Optional[bytes]
    image_path: Optional[str]
    current_activity: str
    apply_activity: bool
    memory_context: str
    user_id: str
    