"""把 CmedqaRetrieval 的 corpus 插入 Milvus（独立完整版）。

流程：
1. 读 corpus parquet（id + text）
2. 建集合（dense + sparse 混合检索字段 + doc_id 字段，无交互，自动重建）
3. 分批插入

关键：schema 里显式建 doc_id 字段，用于检索结果和 qrels 的 pid 对齐算指标。

用法：
    <venv python> insert_cmedqa.py
"""
import pandas as pd

from pymilvus import MilvusClient, DataType, Function, FunctionType, IndexType
from pymilvus.client.types import MetricType
from langchain_milvus import Milvus, BM25BuiltInFunction

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL
from utils.log_utils import log

CORPUS_PATH = '../data/from_modelscope/corpus-00000-of-00001-a3949861f65a3226.parquet'
BATCH_SIZE = 500
LIMIT = 1000  # 测试阶段只插前 1000 条；全量插入改为 None


def build_schema(client: MilvusClient):
    """建集合：dense(向量) + sparse(BM25) + doc_id(用于对齐qrels)。无交互，自动重建。"""
    # 已存在则删掉重建
    if COLLECTION_NAME in client.list_collections():
        log.warning(f"集合 {COLLECTION_NAME} 已存在，自动删除重建")
        client.release_collection(collection_name=COLLECTION_NAME)
        client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
        client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema()
    schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000,
                     enable_analyzer=True,
                     analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
    # 关键：doc_id 用于和 qrels 的 pid 对齐
    schema.add_field(field_name='doc_id', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
    schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=1024)

    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    index_params.add_index(
        field_name="dense",
        index_name="dense_inverted_index",
        index_type=IndexType.HNSW,
        metric_type=MetricType.IP,
        params={"M": 16, "efConstruction": 64},
    )

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)
    log.info(f"集合 {COLLECTION_NAME} 创建完成")


def load_corpus():
    df = pd.read_parquet(CORPUS_PATH)
    if LIMIT:
        df = df.head(LIMIT)
        log.info(f"测试模式：只加载前 {LIMIT} 条")
    docs = []
    for _, row in df.iterrows():
        docs.append({
            'text': row['text'],
            'doc_id': row['id'],
        })
    log.info(f"加载 corpus 完成，共 {len(docs)} 条")
    return docs


def main():
    # 1. 建集合
    client = MilvusClient(uri=MV_URL)
    build_schema(client)

    # 2. langchain-milvus 连接（用于 add_documents，自动生成 dense + sparse）
    mv = Milvus(
        collection_name=COLLECTION_NAME,
        vector_field=['dense', 'sparse'],
        embedding_function=bge_embedding,
        builtin_function=BM25BuiltInFunction(),
        auto_id=True,
        connection_args={'uri': MV_URL},
    )

    # 3. 分批插入
    docs = load_corpus()
    total = 0
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        # add_documents 需要 list[Document] 或 list[dict]? 用 dict+text 需确认，
        # 这里用 langchain Document 更稳
        from langchain_core.documents import Document
        batch_docs = [Document(page_content=d['text'], metadata={'doc_id': d['doc_id']}) for d in batch]
        mv.add_documents(batch_docs)
        total += len(batch)
        log.info(f"已插入 {total}/{len(docs)}")

    log.info(f"插入完成，共 {total} 条")


if __name__ == '__main__':
    main()

