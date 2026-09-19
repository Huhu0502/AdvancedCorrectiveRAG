from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from correctiveRAG.control.fallback import Degradation
from correctiveRAG.message_tool import get_last_human_message, get_last_tool_message
from correctiveRAG.state.state import State
from model.embedding_models import llm_qwen
from utils.log_utils import log

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名语义理解专家，你的任务是润色工具回答，生成简洁、逻辑的话语，'
               '下面是用户的提问:\n {user_question} \n'
               '下面是工具的输出:\n {tool_output}')
])

generate_agent = prompt | llm_qwen


def generate_process(state: State):
    level = state.get('degradation_level', 0)
    if level > Degradation.STATIC:
        log.warn(f'当前处于"generate_node"-->降级等级为{level}')
    else:
        log.info(f'当前处于"generate_node" 未出现降级')

    messages = state['messages']
    human_messages = get_last_human_message(messages)
    tool_messages = get_last_tool_message(messages)

    resp = generate_agent.invoke(
        {
            'tool_output': tool_messages.content,
            'user_question': human_messages
        }
    )
    return {'messages': [resp]}
