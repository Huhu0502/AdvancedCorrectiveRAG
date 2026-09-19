from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage


def get_last_human_message(messages: list[BaseMessage]):
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            return msg
    return None


def get_last_tool_message(messages: list[BaseMessage]):
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            return msg
    return None
