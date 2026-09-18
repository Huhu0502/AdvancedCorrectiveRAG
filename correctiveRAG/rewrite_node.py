from langchain_core.prompts import ChatPromptTemplate

from model.embedding_models import llm_qwen

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名语义理解专家，你的任务是理解用户提问： \n{user_question}'
               '\n，但是这条提问没有很好的效果，你只要输出一条新的提问，替代用户的问题，'
               '除此之外不要回答其他内容')
])

rewriter_agent = prompt | llm_qwen
