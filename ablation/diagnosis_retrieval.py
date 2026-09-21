"""检索质量诊断脚本 —— 用 pymilvus 带 score 的 search

作用：对测试 query 做混合检索（dense + sparse → RRF），打印每条结果的距离分数、
来源文件名、内容摘要。以此判断：
  1. 相关文档 vs 无关文档的分数差距（看 score_threshold 该设多少）
  2. 数据是否主题混杂（看 source 文件名）

用法：
    <venv python> diagnosis_retrieval.py
"""
import pandas as pd
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL


def diagnose(query: str, client, top_k=5):
    print(f"\n{'='*70}")
    print(f"query: {query}")
    print(f"{'='*70}")

    mv = Milvus(
        collection_name=COLLECTION_NAME,
        vector_field=['dense', 'sparse'],
        embedding_function=bge_embedding,
        builtin_function=BM25BuiltInFunction(),
        consistency_level="Strong",
        connection_args={'uri': MV_URL}
    )

    res = mv.similarity_search_with_score(
        query=query,
        k=5,
        expr='category=="content"'
    )

    if not res or not res[0]:
        print("  (无结果)")
        return

    for doc, score in res:
        print(score, '|', doc.metadata.get('source'), '|', doc.page_content[:60])


def main():
    client = MilvusClient(uri=MV_URL)
    # 确认集合存在
    cols = client.list_collections()
    print(f"现有 collections: {cols}")
    if COLLECTION_NAME not in cols:
        print(f"!! 集合 {COLLECTION_NAME} 不存在，请先入库")
        return

    queries = [
        "pi-agent 有什么不足",
        "pi-agent 的架构",
        "等离子体如何改善半导体表面",
        "SDK 怎么使用",
    ]

    for q in queries:
        try:
            diagnose(q, client)
        except Exception as e:
            print(f"\n!!! 检索 {q!r} 出错: {type(e).__name__}: {e}")


if __name__ == '__main__':
    df = pd.read_parquet("../data/from_modelscope/corpus-00000-of-00001-a3949861f65a3226.parquet")
    print(df.head())
    print(df.columns)

    df = pd.read_parquet("../data/from_modelscope/queries-00000-of-00001-daeedab899d3c839.parquet")
    print(df.head())
    print(df.columns)

    df = pd.read_parquet("../data/from_modelscope/dev-00000-of-00001-57fb84a4aceaa695.parquet")
    print(df.head())
    print(df.columns)

