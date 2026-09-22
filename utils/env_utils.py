from dotenv import load_dotenv

load_dotenv(override=True)

COLLECTION_NAME = 'CmedqaRetrieval'
MV_URL = 'http://192.168.127.131:19530'

MAX_REWRITE_COUNT = 3

SENSITIVE_WORDS = ['政治', '侵犯', '盗窃', '违法', '机密']

# 检索调优结论（见 docs/检索调优总结.md）
# 最优配置：base embedding + dense 检索 + score_threshold 过滤
# 阈值 0.65 是 Cmedqa 数据上的最优值，pi-agent 数据需重新探查
DENSE_SCORE_THRESHOLD = 0.65


