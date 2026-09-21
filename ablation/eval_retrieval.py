"""CmedqaRetrieval 检索评估脚本 —— 三种检索方式 ablation 对比

对抽样 query 分别用：
  1. 纯 dense（bge 向量）
  2. 纯 sparse（BM25）
  3. 混合（dense + sparse + RRF）

算 Recall@k / MRR，对比哪种方式召回更好。

用法：
    <venv python> eval_retrieval.py
"""
import random
import pandas as pd
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL

CORPUS_PATH = '../data/from_modelscope/corpus-00000-of-00001-a3949861f65a3226.parquet'
QUERIES_PATH = '../data/from_modelscope/queries-00000-of-00001-daeedab899d3c839.parquet'
DEV_PATH = '../data/from_modelscope/dev-00000-of-00001-57fb84a4aceaa695.parquet'

SAMPLE_N = 100      # 抽样 query 数
K = 5               # top-k
INSERTED_N = 1000   # 已插入的 corpus 条数

def load_valid_queries():
    """返回正样本 pid 全部落在已插入集合里的 query。"""
    corpus = pd.read_parquet(CORPUS_PATH)
    dev = pd.read_parquet(DEV_PATH)
    queries = pd.read_parquet(QUERIES_PATH)

    inserted_ids = set(corpus['id'].head(INSERTED_N).tolist())

    # 每个 qid 的正样本 pid 列表
    qid2pids = dev.groupby('qid')['pid'].apply(list).to_dict()

    # 过滤：正样本 pid 全在库里的 qid
    valid_qids = [q for q, pids in qid2pids.items() if all(p in inserted_ids for p in pids)]

    # 抽样
    random.seed(42)
    sample_qids = random.sample(valid_qids, min(SAMPLE_N, len(valid_qids)))

    qid2text = dict(zip(queries['id'], queries['text']))
    result = []
    for qid in sample_qids:
        result.append({
            'qid': qid,
            'query': qid2text.get(qid, ''),
            'relevant_pids': set(qid2pids[qid]),   # 正样本文档
        })
    return result


def search_dense(client, query, k):
    vec = bge_embedding.embed_query(query)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vec],
        anns_field='dense',
        limit=k,
        search_params={'metric_type': 'IP', 'params': {'nprobe': 16}},
        output_fields=['doc_id'],
    )
    return [hit['entity'].get('doc_id') for hit in res[0]]


def search_sparse(client, query, k):
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[query],
        anns_field='sparse',
        limit=k,
        search_params={'metric_type': 'BM25'},
        output_fields=['doc_id'],
    )
    return [hit['entity'].get('doc_id') for hit in res[0]]


def search_hybrid(client, query, k):
    dense_vec = bge_embedding.embed_query(query)
    req1 = AnnSearchRequest(data=[dense_vec], anns_field='dense',
                            param={'metric_type': 'IP', 'params': {'nprobe': 16}}, limit=k)
    req2 = AnnSearchRequest(data=[query], anns_field='sparse',
                            param={'metric_type': 'BM25'}, limit=k)
    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req1, req2],
        ranker=RRFRanker(100),
        limit=k,
        output_fields=['doc_id'],
    )
    return [hit['entity'].get('doc_id') for hit in res[0]]


def recall_at_k(retrieved, relevant, k):
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant) if relevant else 0.0


def mrr(retrieved, relevant):
    for i, pid in enumerate(retrieved):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def eval_method(name, search_fn, client, queries_data):
    total_recall = 0.0
    total_mrr = 0.0
    n = len(queries_data)
    for item in queries_data:
        retrieved = search_fn(client, item['query'], K)
        total_recall += recall_at_k(retrieved, item['relevant_pids'], K)
        total_mrr += mrr(retrieved, item['relevant_pids'])
    return {
        'method': name,
        f'recall@{K}': round(total_recall / n, 4),
        'mrr': round(total_mrr / n, 4),
    }


def main():
    client = MilvusClient(uri=MV_URL)
    queries_data = load_valid_queries()
    print(f"评估 query 数: {len(queries_data)}")
    print(f"top-k = {K}\n")

    results = [
        eval_method('dense(bge)', search_dense, client, queries_data),
        eval_method('sparse(BM25)', search_sparse, client, queries_data),
        eval_method('hybrid(RRF)', search_hybrid, client, queries_data),
    ]

    print("=" * 40)
    print(f"{'方法':<18}{'Recall@k':<12}{'MRR':<10}")
    print("=" * 40)
    for r in results:
        print(f"{r['method']:<18}{r[f'recall@{K}']:<12}{r['mrr']:<10}")
    print("=" * 40)


if __name__ == '__main__':
    main()
