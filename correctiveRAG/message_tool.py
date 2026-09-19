from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage


def get_last_human_message(messages: list[BaseMessage]):
    human_messages = {}

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            human_messages = msg
            break
    return human_messages


def get_last_tool_message(messages: list[BaseMessage]):
    tool_message = {}

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            human_messages = msg
            break
    return tool_message
