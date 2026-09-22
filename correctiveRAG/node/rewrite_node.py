from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from correctiveRAG.control.fallback import Degradation
from correctiveRAG.message_tool import get_last_human_message
from correctiveRAG.observability.trace import trace_node
from correctiveRAG.state.state import State
from model.embedding_models import llm_qwen
from utils.log_utils import log

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名医疗问诊语义理解专家。请把用户的口语化问题改写成一条规范、简洁、适合检索的医疗查询。\n\n'
               '改写规则：\n'
               '1. 去除口语化前缀和无关信息（如"想确定一下""前几天""怎么回事啊"等），只保留核心医疗诉求\n'
               '2. 把口语症状描述规范化为医学术语（如"红点点"→"红疹"，"长水泡"→"皮肤水疱"）\n'
               '3. 明确用户的核心意图：是问病因、诊断、用药、还是日常护理\n'
               '4. 输出一句话的规范查询，不要超过 30 字\n'
               '5. 不要回答用户，只输出改写后的查询\n\n'
               '用户原话：{user_question}\n'
               '改写后的查询：')
])

rewriter_agent = prompt | llm_qwen


# 改写并不保证检索结果变好。如果改写后的 query 还是检索不到相关内容，judge 又判 no，就会再次改写 → 再次检索，无限绕圈。
#  这就是死循环风险。
# 主要思路：三道防线，从软到硬
#    防线1（逻辑层）：改写次数上限
#         "最多改写 3 次，不再改写，降级直接用现有结果生成"
#
#    防线2（质量层）：重复改写检测
#         "新 query 和历史 query 太像（相似度>0.9），说明改写没用，别写了"
#
#    防线3（工程层）：recursion_limit 硬顶
#         LangGraph 的兜底，即使前两道没拦住，也强制中断
@trace_node("rewrite_node")
def rewriter_process(state: State):
    log.info('当前处于"rewriter_node"')
    messages = state['messages']
    human_messages = get_last_human_message(messages)

    # msg.content  才是回答内容
    log.info(f'human_msg:   {human_messages.content}')
    resp = rewriter_agent.invoke({'user_question': human_messages.content})
    new_query = resp.content

    rewriter_history = state['rewritten_queries']
    for old_query in rewriter_history:
        ratio = SequenceMatcher(None, new_query, old_query).ratio()
        if ratio >= 0.9:
            return {
                'next_step': 'generate',
                'degradation_level': Degradation.NO_REWRITE,
                'rewrite_count': state['rewrite_count'] + 1,
                'control_events': ["new rewrite query's similarity >= 0.9, stop rewriting"]
            }

    return {
        'messages': [HumanMessage(content=resp.content)],
        'rewrite_count': (state['rewrite_count'] or 0) + 1,
        'rewritten_queries': [new_query],
        'next_step': 'main_agent',
        'control_events': ["new query similarity in legal range, allow to rewrite"]
    }
