class Degradation:
    NORMAL = 0  # 相信llm
    NO_REWRITE = 1  # 最后一次相信llm，不再重写
    RETRIEVAL_ONLY = 2  # 不相信llm，只用检索结果
    STATIC = 3  # 完全不相信llm，使用自己的兜底文本


class FallbackPolicy:
    # 降级策略
    @staticmethod
    def build(level, source=None) -> str:
        if level == Degradation.RETRIEVAL_ONLY:
            if source:
                return "(降级：只返回检索结果) \n" + source
            return "抱歉，知识库中暂未找到与您问题直接相关的内容。"
        return "抱歉，我暂时无法给出可靠回答，请稍后重试或换个问法。"
