from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool, Tool
from langchain_milvus import Milvus, BM25BuiltInFunction

from correctiveRAG.observability.trace import trace_node
from correctiveRAG.state.state import State
from model.embedding_models import bge_embedding, llm_qwen
from utils.env_utils import COLLECTION_NAME, MV_URL
from utils.log_utils import log


def get_retriever() -> Tool:
    vector_saved = Milvus(
        collection_name=COLLECTION_NAME,
        vector_field=['dense', 'sparse'],
        builtin_function=BM25BuiltInFunction(),
        embedding_function=bge_embedding,
        auto_id=True,
        connection_args={'uri': MV_URL}
    )

    retriever = vector_saved.as_retriever(
        search_type='similarity',
        search_kwargs={
            "k": 3,
            "score_threshold": 0.1,
            "ranker_type": "rrf",
            "ranker_params": {"k": 100},
            'filter': {"category": "content"}
        }
    )

    return create_retriever_tool(
        retriever,
        name='rag_retriever',
        description='搜索并返回关于"pi-agent的信息"'  # 工具描述，给大模型看的
    )


prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名智能问答助手，擅长使用工具回答问题，你的任务是读取上下文，'
               '。回答任何问题前，必须先调用 rag_retriever 工具检索知识库，'
               '基于检索结果回答。即使你认为自己知道答案，也必须先检索，不得直接回答。'
               '如果问题出现了“agent”、“pi”等专业知识必须调用工具查向量库再回答用户的问题，不能自己编写答案。'
               '这种问题是非常严谨、重要的问题，每次被问到，都要去用工具查询，不能用自己的理解'
               '除非用户输入是纯闲聊（如"你好""谢谢"），否则必须调用 rag_retriever 工具检索知识库后再回答'
               '。一定要问一次查一次！'),
    MessagesPlaceholder(variable_name='messages')
])

retriever_tool = get_retriever()
agent = prompt | llm_qwen.bind_tools([retriever_tool])  # 逻辑绑定 仅绑定不会用 还要在图中物理绑定


# create_react_agent(llm, [tool1,tool2,..]).invoke()可以调工具，因为内部构建了一个 LangGraph 状态图


# 在函数节点内、外定义智能体有什么区别
# 推荐在函数外部定义智能体（Chain/Runnable），节点函数内只做“读 State → 传参调用 → 返回更新”。 两种写法都能跑通，但在 LangGraph 里有明显差异：
# 外部定义（推荐）

# # 模块级：只构建一次，全局复用
# generate_agent = generate_prompt | llm | StrOutputParser()
#
# def generate_node(state: State):
#     tool_output = state['messages'][-1].content
#     human_question = state['messages'][0].content
#     resp = generate_agent.invoke({
#         'human_question': human_question,
#         'tool_output': tool_output
#     })
#     return {'messages': [AIMessage(content=resp)]}
# 性能更优：避免每次节点执行都重复构建 RunnableSequence（| 管道本身是惰性声明，但重复构造仍有开销）。
# 可观测性更好：LangSmith 追踪时，链作为一个独立组件出现，调试时能清晰看到 Prompt 填充、模型输出、解析器处理的完整链路。
# 可复用：同一个 agent 可以被多个节点、测试脚本、甚至其他图共享。
# 职责清晰：节点函数只负责“状态转换逻辑”，agent 负责“LLM 推理逻辑”，符合 LangGraph 节点 = 纯函数的设计哲学。
# 内部定义（不推荐）
# 编辑
# def generate_node(state: State):
#     # ❌ 每次调用都重新构建链
#     agent = prompt | llm | StrOutputParser()
#     resp = agent.invoke({...})
#     return {'messages': [AIMessage(content=resp)]}
# 重复构建开销：每次节点执行都会重新跑一遍 | 运算符，创建新的 RunnableSequence 对象。
# 无法独立测试：想单独测试 agent 的 Prompt 效果时，必须把整个节点函数跑起来，耦合度高。
# LangSmith 追踪混乱：每次调用都生成一个新的链实例，追踪面板里会出现大量重复条目，难以对比分析。
@trace_node("main_agent")
def main_agent_process(state: State):
    resp = agent.invoke({'messages': state['messages']})
    # Agent 节点（LLM 调用）产生的回答结果本身就是 AIMessage 类（或其子类）
    log.info('当前处于"main_agent"')
    log.info(f"main_agent 返回，tool_calls 数量: {len(resp.tool_calls)}")
    return {
        # ' messages '定义是接收Message列表，而不是resp
        'messages': [resp]
    }


if __name__ == '__main__':
    # 临时调试：手动测试 retriever 是否正常
    try:
        docs = retriever_tool.invoke("等离子体清洗 半导体")
        print(f"✅ 检索成功，返回 {len(docs)} 条文档")
        for doc in docs:
            print(doc)
    except Exception as e:
        print(f"❌ 检索失败: {type(e).__name__}: {e}")
