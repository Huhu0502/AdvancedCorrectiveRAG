from langchain_core.prompts import ChatPromptTemplate

from model.embedding_models import llm_qwen

prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一名语义理解专家，你的任务是润色工具回答，生成简洁、逻辑的话语，'
               '下面是用户的提问:\n {user_question} \n'
               '下面是工具的输出:\n {tool_output}')
])

generate_agent = prompt | llm_qwen
