from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from correctiveRAG.control.fallback import Degradation
from correctiveRAG.message_tool import get_last_human_message, get_last_tool_message
from correctiveRAG.observability.trace import trace_node
from correctiveRAG.state.state import State
from model.embedding_models import llm_qwen
from utils.log_utils import log

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名问答助手。请严格仅根据下面提供的上下文回答用户问题。\n'
               '规则：\n'
               '1. 只能用上下文中的信息回答，禁止使用你自己的知识\n'
               '2. 如果上下文能回答部分问题，就基于上下文给出这部分回答，'
               '不要添加任何关于"上下文未涉及"的元评论。'
               '3. 如果上下文完全无法回答，直接说"检索内容不足"，不要展开说明\n'
               '4. 禁止编造上下文中完全不存在的事实\n\n'
               '用户问题：{user_question}\n'
               '上下文：\n{tool_output}')
])

generate_agent = prompt | llm_qwen


@trace_node("generate_node")
def generate_process(state: State):
    level = state.get('degradation_level', 0)
    if level > Degradation.NORMAL:
        log.warning(f'当前处于"generate_node"-->降级等级为{level}')
    else:
        log.info(f'当前处于"generate_node" 未出现降级')

    messages = state['messages']
    human_messages = get_last_human_message(messages)
    tool_messages = get_last_tool_message(messages)

    resp = generate_agent.invoke(
        {
            'tool_output': tool_messages.content,
            'user_question': human_messages.content
        }
    )
    return {'messages': [resp]}
