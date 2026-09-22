from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool, Tool
from langchain_milvus import Milvus, BM25BuiltInFunction

from correctiveRAG.observability.trace import trace_node
from correctiveRAG.state.state import State
from model.embedding_models import bge_embedding, llm_qwen
from utils.env_utils import COLLECTION_NAME, MV_URL, DENSE_SCORE_THRESHOLD
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

    # 检索方案：最优配置来自 ablation 调优结论（见 docs/检索调优总结.md）
    # 核心：dense 单路检索 + score_threshold过滤（比 RRF 混合更优）
    # 阈值 0.65 是 Cmedqa 数据上的最优值，pi-agent 数据需重新探查
    retriever = vector_saved.as_retriever(
        search_type='similarity',
        search_kwargs={
            "k": 7,
            # 注意：不再用 rrf 混合检索（调优结论：hybrid 无益）
        }
    )

    return create_retriever_tool(
        retriever,
        name='rag_retriever',
        description='搜索并返回关于"医疗、个人健康问题等信息"'  # 工具描述，给大模型看的
    )


prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名医疗问答助手。回答专业问题前，必须先调用 rag_retriever 工具检索知识库，'
               '严格基于检索结果回答。\n'
               '规则：\n'
               '1. 任何专业问题都必须先调用工具检索，禁止凭自己的知识直接回答\n'
               '2. 只能基于工具返回的内容回答，禁止添加或编造检索结果中没有的信息\n'
               '3. 如果检索结果与问题无关或不充分，如实说明"检索结果不足"，不要牵强推断\n'
               '4. 除非用户输入是纯闲聊（如"你好""谢谢"），否则必须调用工具检索后再回答\n'
               '5. 如果对话中有多条用户消息，以【最后一条用户消息】作为要检索的问题，'
               '用它作为检索工具的 query，忽略之前的消息\n'),
    MessagesPlaceholder(variable_name='messages')
])

retriever_tool = get_retriever()
agent = prompt | llm_qwen.bind_tools([retriever_tool])  # 逻辑绑定 仅绑定不会用 还要在图中物理绑定


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
