from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusClient, DataType, Function, FunctionType, IndexType
from pymilvus.client.types import MetricType

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL


class OperateMilvus:
    def __init__(self):
        self.mv: Milvus = None  # 保存向量库连接

    def create_connection(self):
        self.mv = Milvus(
            collection_name=COLLECTION_NAME,
            vector_field=['dense', 'sparse'],
            embedding_function=bge_embedding,
            builtin_function=BM25BuiltInFunction(),
            consistency_level="Strong",
            auto_id=True,
            connection_args={'uri': MV_URL}
        )

    def create_collection(self):
        client = MilvusClient(uri=MV_URL)
        collections = client.list_collections()

        # 库中存在同名集合则删除
        if COLLECTION_NAME in collections:
            is_del = input(f'集合{COLLECTION_NAME}已存在，即将删除并重建集合，是否执行？ [Y/N]')
            if is_del in ['Y', 'yes', 'YES', 'Yes', 'y']:
                # 先释放， 再删除索引，再删除collection
                client.release_collection(collection_name=COLLECTION_NAME)
                client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
                client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
                client.drop_collection(collection_name=COLLECTION_NAME)
            else:
                return

        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=6000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=1024)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def insert_datas(self, datas: List[Document]):
        self.mv.add_documents(datas)


if __name__ == '__main__':
    tool = OperateMilvus()
    tool.create_collection()
    tool.create_connection()
    tool.insert_datas([
        Document(page_content="示例", metadata={'source': 'xxx', 'page': 1})
    ])

    res = tool.mv.similarity_search(
        query='找一个示例',
        k=1
    )

    for doc in res:
        print(doc)
