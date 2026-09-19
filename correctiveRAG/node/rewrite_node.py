from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from correctiveRAG.state.state import State
from model.embedding_models import llm_qwen
from utils.log_utils import log

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名语义理解专家，你的任务是理解用户提问： \n{user_question}'
               '\n，但是这条提问没有很好的效果，你只要输出一条新的提问，替代用户的问题，'
               '除此之外不要回答其他内容')
])

rewriter_agent = prompt | llm_qwen


def rewriter_process(state: State):
    messages = state['messages']
    human_messages = {}

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            human_messages = msg

    # msg.content  才是回答内容
    log.info(f'human_msg:   {human_messages.content}')
    resp = rewriter_agent.invoke({'user_question': human_messages.content})
    log.info('当前处于"rewriter_node"')
    return {'messages': [HumanMessage(content=resp.content)]}
