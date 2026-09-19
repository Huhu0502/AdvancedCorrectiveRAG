from correctiveRAG.control.fallback import Degradation
from correctiveRAG.state.state import State


def decide_process(state: State) -> dict:
    result = state.get('judge_result')
    if result == "yes":
        return {
            'next_step': 'generate',
            'control_events': ["decide to generate"]
        }
    elif result == "no":
        return {
            'next_step': 'rewrite',
            'control_events': ["decide to rewrite"]
        }
    else:
        return {
            'next_step': 'fallback',
            'degradation_level': Degradation.RETRIEVAL_ONLY,
            'control_events': [f"decide to fallback (unexpected judge_result={result!r})"],
        }

