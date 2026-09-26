from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from langchain_core.runnables import Runnable
from typer import prompt

from ai_companion.core.prompts import (ROUTER_PROMPT, CHARACTER_CARD_PROMPT)
from ai_companion.graph.utils.helpers import (get_chat_model, AsteriskRemovalParser)



class RouterResponse(BaseModel):
    response_type: Literal["conversation", "image", "audio"]
    
    
def get_router_chain() -> Runnable:
    model = get_chat_model(temperature=0.3).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model




def get_character_response_chain(summary: str = "") -> Runnable:
    model = get_chat_model()
    system_message = (f"{CHARACTER_CARD_PROMPT}\n\nSummary..." if summary else CHARACTER_CARD_PROMPT)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | model | AsteriskRemovalParser()