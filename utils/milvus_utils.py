from langchain_milvus import Milvus, BM25BuiltInFunction

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL


class OperateMilvus:
    def __init__(self):
        self.mv: Milvus = None  # 记录向量库连接

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
        pass

    def insert_datas(self, datas):
        self.mv.add_documents(datas)
