from langchain_core.messages import AIMessage

from correctiveRAG.control.fallback import FallbackPolicy
from correctiveRAG.message_tool import get_last_tool_message
from correctiveRAG.observability.trace import trace_node
from correctiveRAG.state.state import State
from utils.log_utils import log


@trace_node("fallback_node")
def fallback_process(state: State):
    level = state['degradation_level']
    log.warning(f"当前处于'fallback_node'-->降级等级为 {level} ")

    tool_msg = get_last_tool_message(state['messages'])
    final_answer = FallbackPolicy.build(level=level, source=tool_msg.content or None)

    return {
        'messages': [AIMessage(content=final_answer)],
        'control_events': [f"fallback L{level}"]
    }


