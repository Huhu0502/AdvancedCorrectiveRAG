import operator
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class State(TypedDict):
    """
    langgraph中的共享数据结构
    """
    messages: Annotated[list[BaseMessage], add_messages]
    judge_result: str  # 'yes', 'no', 'unknown'
    next_step: str  # 'generate', 'rewrite', 'fallback'
    degradation: int  # 0/1/2/3
    control_events: Annotated[list[str], operator.add]


# 数据模型
class Grade(BaseModel):
    """相关性检查的二元评分"""
    binary_score: Literal["yes", "no"] = Field(description="相关性评分 'yes' 或 'no'")
