import os

from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

embedding_model = OpenAIEmbeddings(
    base_url=os.getenv('dashscope_base_url'),
    api_key=os.getenv('dashscope_apikey'),
    model='qwen3.7-text-embedding'
)

embedding = OllamaEmbeddings(
    model='nomic-embed-text:latest',
    base_url='http://localhost:11434'
)

model_name = 'BAAI/bge-small-zh-v1.5'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}  # 归一化一定要开
bge_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

llm_qwen = ChatOpenAI(
    base_url=os.getenv('dashscope_base_url'),
    model=os.getenv('dashscope_chat_options_model'),
    api_key=os.getenv('dashscope_apikey')
)

llm_ollama_judge = ChatOpenAI(
    base_url='http://localhost:11434/v1',
    model='qwen2.5:7b-instruct',
    api_key='ollama'
)
