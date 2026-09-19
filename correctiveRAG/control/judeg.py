import re

from langchain_core.prompts import ChatPromptTemplate

from correctiveRAG.state.state import Grade
from model.embedding_models import deepseek_llm
from utils.log_utils import log


def judge(question, context) -> tuple[str, str]:
    prompt = ChatPromptTemplate.from_template("你是一名语义理解助手。\n"
                                              "请判断以下用户问题与检索到的回答之间是否语义相关。\n\n"
                                              "用户问题：{human_question}\n"
                                              "检索回答：{tool_msg}\n"
                                              "你只要返回  yes  或者  no \n"
                                              "不需要返回除此之外的任何内容！")
    # 一定要打印日志！！！有时候出错不一定报错
    # 部分不支持结构化输出，导致返回不可控制的结果，最后路由失败报错。with_structured_output 已经自动解析并返回对象，
    # invoke() 的返回值不再是 AIMessage，没有 .content 属性。
    judge_agent = prompt | deepseek_llm.with_structured_output(Grade)
    raw_text = ""
    try:  # 网络错误、DeepSeek 不支持函数调用、解析失败
        raw = judge_agent.invoke(input={'human_question': question, 'tool_msg': context})
        raw_text = raw.binary_score or " "  # 防止binary_score为None
        decision = raw_text.strip().lower()
        #  第一层正确返回
        if decision in ("yes", "no"):
            log.info(f"judge结构化判断成功 {decision}")
            return decision, raw_text
    except Exception as e:
        log.warning(f"结构化判断失败，进入正则兜底: {type(e).__name__}: {e}")

    # 第二层自由回答＋正则兜底
    plant_agent = prompt | deepseek_llm
    try:
        msg = plant_agent.invoke(input={'human_question': question, 'tool_msg': context})
        raw_text = str(msg.content or "")
        decision = _regex_match(raw_text)
        if decision:
            return decision, raw_text
    except Exception as e:
        log.warning(f"判断模型调用失败: {type(e).__name__}: {e}")

    # 第三层unknown返回
    log.warning(f"judge 无法解析输出: {raw_text!r}")
    return "unknown", raw_text


def _regex_match(text: str):
    t = (text or "").strip().lower()
    if re.search(r"\bno\b否|不相关|无关|不相干", t):
        return "no"
    if re.search(r"\byes\b|是|相关|有关", t):
        return "yes"
    return None
