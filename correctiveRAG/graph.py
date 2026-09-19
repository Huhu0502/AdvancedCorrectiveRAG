import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition

from correctiveRAG.control.fallback import Degradation
from correctiveRAG.node.agent_node import retriever_tool, main_agent_process
from correctiveRAG.node.decision_node import decide_process
from correctiveRAG.node.fallback_node import fallback_process
from correctiveRAG.node.generate_node import generate_process
from correctiveRAG.node.judge_node import judge_process
from correctiveRAG.node.rewrite_node import rewriter_process
from correctiveRAG.state.state import State
from correctiveRAG.control.tool_handler import create_tool_node_with_fallback
from utils.draw_graph import draw_graph
from utils.log_utils import log

builder = StateGraph(State)

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

builder.add_node('judge_node', judge_process)
builder.add_node('decide_node', decide_process)
builder.add_node('fallback_node', fallback_process)

builder.add_edge('fallback_node', END)
builder.add_edge('retriever_tool', 'judge_node')
builder.add_edge('judge_node', 'decide_node')


def decide_router(state: State):
    next_step = state.get('next_step')
    if next_step:
        return next_step

    log.error(f"状态数据next_step异常: {next_step}, 流程结束")
    return END


builder.add_conditional_edges(
    'decide_node',
    decide_router,
    {
        'generate': 'generate_node',
        'rewrite': 'rewriter_node',
        'fallback': 'fallback_node',
        END: END
    }
)

builder.add_node('rewriter_node', rewriter_process)
builder.add_edge('rewriter_node', 'main_agent')

builder.add_node('generate_node', generate_process)
builder.add_edge('generate_node', END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# draw_graph(graph, 'graph2.png')
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
            initial_state = {
                'messages': [('user', question)],
                'judge_result': '',
                'next_step': '',
                'degradation': Degradation.NORMAL,  # 0
                'control_events': [],
            }
            # 【第 1 步】瞬间完成。Python 只是创建了一个生成器对象赋给 events，图还没开始跑。
            #  stream_mode=value 每个节点跑完后把整个state快照吐出来一次，结合打印部分代码，每个跑完打印最后一条消息，
            #  那么在judge_node和decide_node没有新增消息，就把之前的最后一条消息也就是tool msg打印，所以一个没有重写的trace里打印了3次
            events = graph.stream(input=initial_state, config=config, stream_mode='updates')
            # 【第 2 步】图开始跑！
            # 当 Python 执行到 for 循环的第一次迭代（第一次执行 print）时，
            # 它会向生成器“要”第一个值。此时，LangGraph 引擎才会被唤醒，
            # 开始执行图中的第一个节点（比如 get_user_info），执行完后 yield 一个 event。
            for event in events:
                for node_name, node_output in event.items():
                    msgs = node_output.get("messages", [])
                    if msgs:
                        for m in msgs:
                            print(m.pretty_repr(html=True))
            print("[DEBUG] 警告：图已执行完毕或中断失败，next 为空！")
            # 执行用户下一句对话
