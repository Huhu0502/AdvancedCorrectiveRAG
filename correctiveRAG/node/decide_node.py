from correctiveRAG.control.fallback import Degradation
from correctiveRAG.control.loop_guard import allow_rewrite
from correctiveRAG.state.state import State
from utils.env_utils import MAX_REWRITE_COUNT
from utils.log_utils import log


def decide_process(state: State) -> dict:
    log.info('当前处于"decide_node"')
    result = state.get('judge_result')
    if result == "yes":  # 可以生成答案
        return {
            'next_step': 'generate',
            'control_events': ["decide to generate"]
        }
    elif result == "no":  # 结果不满意，重写问题
        rewrite_count = state.get('rewrite_count', 0)
        ok, reason = allow_rewrite(rewrite_count)
        if ok:
            return {
                'next_step': 'rewrite',
                'control_events': [f"decide to rewrite ({reason})"]
            }
        else:  # 重写次数过多，不再重写，降级处理
            return {
                'next_step': 'generate',
                'degradation_level': Degradation.NO_REWRITE,
                'control_events': [f"decide to generate ({reason})"],
            }
    else:  # 异常、降级处理
        return {
            'next_step': 'fallback',
            'degradation_level': Degradation.RETRIEVAL_ONLY,
            'control_events': [f"decide to fallback (unexpected judge_result={result!r})"],
        }

