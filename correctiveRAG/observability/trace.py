import functools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from correctiveRAG.control.fallback import Degradation
from correctiveRAG.observability.context.trace_context import TraceContext


def _safe_text(v, max_len=300):
    """把任意值安全转成截断后的字符串"""
    if v is None:
        return None
    if isinstance(v, str):
        return v[:max_len]
    try:
        return str(v)[:max_len]
    except Exception:
        return "<unserializable>"


@dataclass
class Span:
    name: str  # 当前节点名
    kind: str = 'node'  # 节点类型 node/llm/tool/router/fallback

    start_at: float = field(default_factory=time.time)  # 开始时间戳
    end_at: float = None  # 结束时间戳
    latency_ms: float = None  # 总耗时

    input: Any = None  # 单元的输入 （节点收到什么、LLM 的 prompt、工具的 query） 不同地方可能传不同东西（节点传 dict、LLM 传 str、工具传 str）
    output: Any = None  # 单元输出  （节点返回什么、LLM 生成什么、工具的结果）

    error: str = None  # 记录异常信息
    meta: dict = field(default_factory=dict)  # 放 kind 相关的额外信息，避免为每种 kind 建一堆字段

    def finish(self, output: str = None, error: str = None, meta: dict = None):
        self.end_at = time.time()
        self.latency_ms = round((self.end_at - self.start_at) * 1000, 2)
        if output is not None:
            self.output = output
        if error is not None:
            self.error = error
        if meta is not None:
            self.meta.update(meta)
        return self

    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "started_at": self.start_at,
            "ended_at": self.end_at,
            "latency_ms": self.latency_ms,
            "input": _safe_text(self.input),  # Any --> str, 还要防止过长
            "output": _safe_text(self.output),
            "error": self.error,
            "meta": self.meta,  # meta 本身是 dict，直接放
        }


@dataclass
class Trace:
    session_id: str = ""  # 同一个用户的会话
    turn_id: int = 0  # 多轮对话的轮序
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # trace唯一id

    user_input: str = ""
    final_answer: str = None

    spans: list[Span] = field(default_factory=list)  # 核心，这一轮所有节点、tool、llm的span列表，记录完整调用链

    status: str = "running"  # 记录这一轮的结果状态： running -> success/error/block/degraded
    degradation_level: int = Degradation.NORMAL  # 如果degraded，降到哪一级(0/1/2/3)
    error: str = None  # 如果出错，出错内容

    start_at: float = field(default_factory=time.time)  # 开始时间戳
    end_at: float = None  # 结束时间戳
    latency_ms: float = None  # 总耗时

    def add_span(self, span: Span):
        self.spans.append(span)
        return span

    def finish(self, final_answer: str = None, status: str = None, degradation_level: int = None,
               error: str = None):
        self.end_at = time.time()
        if final_answer is not None:
            self.final_answer = final_answer
        if status is not None:
            self.status = status
        if degradation_level is not None:
            self.degradation_level = degradation_level
        if error is not None:
            self.error = error
        self.latency_ms = round((self.end_at - self.start_at) * 1000, 2)

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "final_answer": self.final_answer,
            "spans": [s.to_dict() for s in self.spans],  # 逐个转
            "status": self.status,
            "degradation_level": self.degradation_level,
            "error": self.error,
            "started_at": self.start_at,
            "ended_at": self.end_at,
            "latency_ms": self.latency_ms,
        }


def _summarize_messages(messages, max_len=300):
    """把消息列表转成 [{'role':..., 'content':...}, ...] 的可读形式"""
    if messages is None:
        return None
    result = []
    for m in messages:
        role = getattr(m, 'type', '?')  # human/ai/tool
        content = getattr(m, 'content', '')
        # content 可能是 str，也可能是 list（多模态）
        if isinstance(content, list):
            content = ' '.join(
                str(c.get('text', '')) if isinstance(c, dict) else str(c)
                for c in content
            )
        result.append({
            'role': role,
            'content': _safe_text(content, max_len),  # 截断
        })
    return result


def _summary(result: dict, max_len=300):
    if result is None:
        return None
    # 如果是 dict（节点返回的都是 dict）
    if isinstance(result, dict):
        out = {}
        for k, v in result.items():
            if k == 'messages':
                # messages 是列表，只取每个消息的关键内容
                out['messages'] = _summarize_messages(v, max_len)
            elif k == 'control_events':
                # 日志列表，直接保留（本身简短）
                out['control_events'] = _safe_text(v, max_len)
            else:
                out[k] = _safe_text(v, max_len)
        return out

    # 不是 dict，直接安全转字符串
    return _safe_text(result, max_len)


def trace_node(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            trace = TraceContext.get_trace()
            span = trace.add_span(Span(name=name, kind='node')) if trace else None
            t0 = time.time()
            try:
                res = fn(*args, **kwargs)
                if span:
                    span.finish(output=_summary(res))
                return res
            except Exception as e:
                if span:
                    span.finish(error=str(e))
                raise

        return wrapper

    return deco
