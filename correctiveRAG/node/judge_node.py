from correctiveRAG.control.judge import judge
from correctiveRAG.message_tool import get_last_human_message
from correctiveRAG.state.state import State


def judge_process(state: State):
    human_msg = get_last_human_message(state['messages'])
    tool_msg = state['messages'][-1]
    decision, raw_text = judge(human_msg.content, tool_msg.content)

    return {
        'judge_result': decision,
        'control_events': [f"判断模型judge={decision} raw={raw_text[:50]!r}"]
    }
