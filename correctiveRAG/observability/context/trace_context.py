import contextvars
from contextlib import contextmanager
from typing import Optional


# contextvars 跟随"逻辑上下文"而不是"物理线程"，能正确处理三件 threading.local 处理不好的事：
# ①LangGraph 内部可能的线程池/异步派生 ②FastAPI 协程
#  │ 并发交错 ③嵌套 with 的自动恢复。用 contextvars 是"现在够用 + 未来不返工"的标准选择.
# contextvars属于协程、thread local属于线程，以后使用fastapi，一个请求是一个协程，使用thread local会串
_current_trace: contextvars.ContextVar[Optional["Trace"]] = contextvars.ContextVar("current_trace", default=None)


class TraceContext:
    @staticmethod
    @contextmanager  # 把一个普通函数变成 with 语句能用的上下文管理器
    def activate(trace):
        token = _current_trace.set(trace)  # ：执行 with 时先运行到的第一行。把 _current_trace 设成当前 trace
        try:
            yield trace  # yield trace 意味着 with ... as x 里的 x 就是 trace
        finally:
            _current_trace.reset(token)  # ：with 块执行完（无论正常结束还是抛异常），一定会执行

    @staticmethod
    def get_trace():
        return _current_trace.get()
