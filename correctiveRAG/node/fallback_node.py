from langchain_core.messages import AIMessage

from correctiveRAG.control.fallback import FallbackPolicy
from correctiveRAG.message_tool import get_last_tool_message
from correctiveRAG.state.state import State
from utils.log_utils import log


def fallback_process(state: State):
    log.info('当前处于"fallback_node"')
    level = state['degradation_level']
    tool_msg = get_last_tool_message(state['messages'])
    final_answer = FallbackPolicy.build(level=level, source=tool_msg)

    return {
        'messages': [AIMessage(content=final_answer)],
        'control_events': [f"fallback L{level}"]
    }


