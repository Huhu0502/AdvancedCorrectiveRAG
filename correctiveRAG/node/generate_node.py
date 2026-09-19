from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

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
    messages = state['messages']
    human_messages = {}

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            human_messages = msg
    tool_messages = messages[-1]

    resp = generate_agent.invoke(
        {
            'tool_output': tool_messages.content,
            'user_question': human_messages
        }
    )
    log.info('当前处于"generate_node"')
    return {'messages': [resp]}