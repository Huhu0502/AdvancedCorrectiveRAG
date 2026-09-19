from utils.env_utils import MAX_REWRITE_COUNT


def allow_rewrite(rewrite_count: int) -> tuple[bool, str]:
    if rewrite_count >= MAX_REWRITE_COUNT:
        return False, 'accessed the max rewrite limit, generate the final answer'
    return True, 'allow rewrite'
