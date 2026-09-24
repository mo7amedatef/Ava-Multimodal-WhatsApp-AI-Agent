from ai_companion.settings import settings

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import get_router_chain




async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    return {"workflow": response.response_type}