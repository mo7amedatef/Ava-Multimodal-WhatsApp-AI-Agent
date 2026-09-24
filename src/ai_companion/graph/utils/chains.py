from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel

from ai_companion.core.prompts import ROUTER_PROMPT
from ai_companion.graph.utils.helpers import get_chat_model



class RouterResponse(BaseModel):
    response_type: Literal["conversation", "image", "audio"]
    
    
def get_router_chain():
    model = get_chat_model(temperature=0.3).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model