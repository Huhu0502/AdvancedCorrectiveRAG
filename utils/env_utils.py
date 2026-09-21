from dotenv import load_dotenv

load_dotenv(override=True)

COLLECTION_NAME = 'CmedqaRetrieval'
MV_URL = 'http://192.168.127.131:19530'

MAX_REWRITE_COUNT = 3

SENSITIVE_WORDS = ['政治', '侵犯', '盗窃', '违法', '机密']


