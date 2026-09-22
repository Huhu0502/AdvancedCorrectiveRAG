# 查询改写 prompt：把口语化 query 规范成医疗检索词
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from correctiveRAG.observability.trace import trace_node
from correctiveRAG.state.state import State
from model.embedding_models import llm_qwen
from utils.log_utils import log

rewrite_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名医疗问诊语义理解专家。请把用户的口语化问题改写成一条规范、简洁、适合检索的医疗查询。\n\n'
               '改写规则：\n'
               '1. 去除口语化前缀和无关信息（如"想确定一下""前几天""怎么回事"等），只保留核心医疗诉求\n'
               '2. 把口语症状描述规范化为医学术语（如"红点点"→"肛周红疹"，"长水泡"→"皮肤水疱"）\n'
               '3. 明确用户的核心意图：问病因、诊断、用药、还是日常护理\n'
               '4. 输出一句话的规范查询，不超过 30 字\n'
               '5. 不要回答用户，只输出改写后的查询\n\n'
               '用户原话：{user_question}\n'
               '改写后的查询：')
])

query_rewriter = rewrite_prompt | llm_qwen


@trace_node("query_rewrite_node")
def query_rewrite_process(state: State):
    """入口查询改写：把第一次用户提问规范成检索友好的 query。"""
    log.info('当前处于"query_understand_node"')

    raw_question = state['messages'][0].content
    log.info(f'原始 query: {raw_question}')

    resp = query_rewriter.invoke({'user_question': raw_question})
    normalized = str(resp.content or '').strip() or raw_question  # 兜底：改写失败用原文

    log.info(f'规范化 query: {normalized}')

    # 用规范后的 query 替换第一条 user 消息（用于后续检索）
    replaced_messages = list(state['messages'])
    replaced_messages[0] = HumanMessage(content=normalized)

    return {
        'messages': replaced_messages,  # 替换第一条，避免 main_agent 混淆
        'control_events': [f"query_rewrite: {raw_question[:20]} -> {normalized[:20]}"],
    }
