from langchain_core.callbacks import BaseCallbackHandler

from correctiveRAG.observability.context.trace_context import TraceContext
from correctiveRAG.observability.trace import Span, _safe_text


def _llm_name(serialized: dict) -> str:
    name = serialized.get("name")
    if name:
        return str(name)
    sid = serialized.get("id")
    if isinstance(sid, list) and sid:
        return str(sid[-1])
    if sid:
        return str(sid)
    return "llm"


def _first_text(response, max_len=300):
    try:
        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        text = getattr(msg, "content", "") if msg is not None else getattr(gen, "text", "")
        return _safe_text(text, max_len)
    except Exception:
        return None


class TraceCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self._llm_spans = {}
        self._tool_spans = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        trace = TraceContext.get_trace()
        if trace is None:
            return
        span = trace.add_span(Span(name=_llm_name(serialized), kind="llm"))
        inv = kwargs.get("invocation_params") or {}
        span.meta["model"] = inv.get("model", "?")
        span.input = [_safe_text(p, 300) for p in prompts]
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._llm_spans[run_id] = span

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        span = self._llm_spans.pop(run_id, None) if run_id is not None else None
        if span is None:
            return
        usage = (response.llm_output or {}).get("token_usage") or {}
        span.meta["tokens"] = {k: v for k, v in usage.items() if isinstance(v, (int, float))}
        span.finish(output=_first_text(response, 300))

    def on_tool_start(self, serialized, input_str, **kwargs):
        trace = TraceContext.get_trace()
        if trace is None:
            return
        span = trace.add_span(Span(name=serialized.get("name", "tool"), kind="tool"))
        span.input = _safe_text(input_str, 300)
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._tool_spans[run_id] = span

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        span = self._tool_spans.pop(run_id, None) if run_id is not None else None
        if span is None:
            return
        span.finish(output=_safe_text(output, 300))
