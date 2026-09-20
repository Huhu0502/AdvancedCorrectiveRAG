import re

from utils.env_utils import SENSITIVE_WORDS
from utils.log_utils import log


class SensitiveWordGuard:
    def __init__(self):
        self.words = SENSITIVE_WORDS
        self._pattern = None
        if self.words:
            self._pattern = re.compile(
                "|".join(re.escape(w) for w in self.words),
                re.IGNORECASE
            )

    @property
    def enabled(self):
        return self._pattern is not None

    def check_input(self, text) -> tuple[bool, list[str]]:
        if not self.enabled or not text:
            return True, []
        matched = self._pattern.findall(text)
        ok = len(matched) == 0
        if not ok:
            log.warning(f"敏感词命中，输入被拦截: {matched}")
        return ok, matched

    def mask_output(self, text) -> str:
        if not self.enabled or not text:
            return text
        return self._pattern.sub(lambda m: "*" * len(m.group()), text)
