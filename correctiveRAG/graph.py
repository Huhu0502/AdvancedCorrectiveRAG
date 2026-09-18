import re
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

from correctiveRAG.agent_node import retriever_tool, agent
from correctiveRAG.generate_node import generate_agent
from correctiveRAG.rewrite_node import rewriter_agent
from correctiveRAG.state import State, Grade
from correctiveRAG.tool_handler import create_tool_node_with_fallback
from model.embedding_models import llm_ollama_judge
from utils.log_utils import log

builder = StateGraph(State)


def main_agent_process(state: State):
    resp = agent.invoke(state)
    # Agent 节点（LLM 调用）产生的回答结果本身就是 AIMessage 类（或其子类）
    log.info('当前处于"main_agent"')
    return {
        # ' messages '定义是接收Message列表，而不是resp
        'messages': [resp]
    }


builder.add_node('main_agent', main_agent_process)
builder.add_edge(START, 'main_agent')
builder.add_node('retriever_tool', create_tool_node_with_fallback([retriever_tool]))
builder.add_conditional_edges(
    'main_agent',
    tools_condition,
    {
        'tools': 'retriever_tool',
        END: END
    }
)


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


builder.add_node('rewriter_node', rewriter_process)
builder.add_edge('rewriter_node', 'main_agent')


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


builder.add_node('generate_node', generate_process)
builder.add_edge('generate_node', END)


def rewriter_router(state: State):
    state_msg = state['messages']
    tool_msg = state_msg[-1]
    human_messages = {}

    for i in range(len(state_msg) - 1, -1, -1):
        msg = state_msg[i]
        if isinstance(msg, HumanMessage):
            human_messages = msg

    prompt = ChatPromptTemplate.from_template("你是一名语义理解助手。\n"
                                              "请判断以下用户问题与检索到的回答之间是否语义相关。\n\n"
                                              "用户问题：{human_question}\n"
                                              "检索回答：{tool_msg}\n"
                                              "你只要返回  yes  或者  no \n"
                                              "不需要返回除此之外的任何内容！")
    # 一定要打印日志！！！有时候出错不一定报错
    # 本地模型qwen2.5-7b不支持结构化输出，导致返回不可控制的结果，最后路由失败报错
    judge_agent = prompt | llm_ollama_judge
    raw_content = judge_agent.invoke(input={'human_question': human_messages, 'tool_msg': tool_msg})

    if re.search(r'\bYES\b', raw_content.content.strip().upper()):
        log.info(f'judge_llm 判断结果： {raw_content.content.strip()}')
        return "yes"
    elif re.search(r'\bNO\b', raw_content.content.strip().upper()):
        log.info(f'judge_llm 判断结果： {raw_content.content.strip()}')
        return "no"
    else:
        log.error('judge_llm 返回异常，无法路由，退出程序')
        return END


builder.add_conditional_edges(
    'retriever_tool',
    rewriter_router,
    {'yes': 'generate_node', 'no': 'rewriter_node', END: END}
)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def draw_graph(graph, file_name: str):
    try:
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(file_name, "wb") as f:
            f.write(mermaid_code)

    except Exception as e:
        # 这需要一些额外的依赖项，是可选的 pass
        log.exception(e)


# draw_graph(graph, 'graph1.png')
session_id = str(uuid.uuid4())
# update_dates()  # 每次启动就更新数据库某些时间字段，换成最新时间

config = {
    'configurable': {
        'thread_id': session_id
    },
}

#  这个报错的根本原因是 LangGraph 的 ToolNode（或你自定义的 tool_edge）期望从 State 中读取 messages 字段，但你的 State 中实际使用的键名是 message（少了个 s）。
# LangGraph 的预构建组件（如 ToolNode、tools_condition）以及大多数官方示例都硬编码依赖 messages 作为消息列表的字段名。当它尝试访问 state["messages"] 时找不到该键，就会抛出 No messages found in input state。

if __name__ == '__main__':

    # while True: 里的循环，才是用户多轮对话的推进。
    while True:
        question = input('用户输入：')
        if question in ['q', 'quit', 'exit']:
            break
        else:
            # 【第 1 步】瞬间完成。Python 只是创建了一个生成器对象赋给 events，图还没开始跑。
            events = graph.stream({'messages': ('user', question)}, config, stream_mode='values')
            # 【第 2 步】图开始跑！
            # 当 Python 执行到 for 循环的第一次迭代（第一次执行 print）时，
            # 它会向生成器“要”第一个值。此时，LangGraph 引擎才会被唤醒，
            # 开始执行图中的第一个节点（比如 get_user_info），执行完后 yield 一个 event。
            for event in events:
                messages = event.get('messages')
                msg = messages[-1]
                print(msg.pretty_repr(html=True))

            print("[DEBUG] 警告：图已执行完毕或中断失败，next 为空！")
            # 执行用户下一句对话
