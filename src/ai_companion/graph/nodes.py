import os
from uuid import uuid4
from urllib import response

from ai_companion.settings import settings

from ai_companion.graph.state import AICompanionState
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.graph.utils.chains import (get_router_chain, get_character_response_chain)
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.graph.utils.helpers import (get_text_to_image_module, get_text_to_speech_module)


from langchain_core.messages import (AIMessage, HumanMessage)
from langchain_core.runnables import RunnableConfig




async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""
    if not state["messages"]:
        return {}
    
    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}



async def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    memory_manager = get_memory_manager()

    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}



async def router_node(state: AICompanionState):
    """Route the conversation to the appropriate workflow based on the last messages."""
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    return {"workflow": response.response_type}



async def context_injection_node(state: AICompanionState):
    """Inject Ava's current activity into the state based on the time of day."""
    schedule_context = ScheduleContextGenerator.get_current_activity()
    apply_activity = schedule_context is not None and schedule_context != state.get("current_activity", "")
    return {"apply_activity": apply_activity, "current_activity": schedule_context or ""}



async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}




async def image_node(state: AICompanionState, config: RunnableConfig) -> dict:
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])

    img_path = f"{settings.OUTPUT_DIR}/image_{uuid4()}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    scenario_message = AIMessage(content=f"<image attached: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response), "image_path": img_path}





async def audio_node(state: AICompanionState, config: RunnableConfig) -> dict:
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    output_audio = await text_to_speech_module.synthesize(response)

    return {"messages": AIMessage(content=response), "audio_buffer": output_audio}