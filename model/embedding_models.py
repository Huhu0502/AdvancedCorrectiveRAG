import os

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

# 线上embedding模型
embedding_model = OpenAIEmbeddings(
    base_url=os.getenv('dashscope_base_url'),
    api_key=os.getenv('dashscope_apikey'),
    model='qwen3.7-text-embedding'
)

# 本地embedding模型
embedding = OllamaEmbeddings(
    model=os.getenv('ollama_embedding_model'),
    base_url=os.getenv('ollama_embedding_base_url')
)


# 向量库查询时使用的bge模型，归一化一定要开
bge_embedding = HuggingFaceEmbeddings(
    model_name='D:/huggingface_cache/models/BAAI--bge-large-zh-v1.5/snapshots/master',   # 本地绝对路径（768维）
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

llm_qwen = ChatOpenAI(
    base_url=os.getenv('dashscope_base_url'),
    model=os.getenv('dashscope_chat_options_model'),
    api_key=os.getenv('dashscope_apikey')
)

deepseek_llm = ChatOpenAI(
    base_url=os.getenv('deepseek_base_url'),
    model=os.getenv('deepseek_model'),
    api_key=os.getenv('deepseek_api_key')
)
