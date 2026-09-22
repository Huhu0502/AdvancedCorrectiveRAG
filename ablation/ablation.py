"""CmedqaRetrieval 消融实验框架

支持多组消融对象对比：
  1. 检索方式：dense / sparse / hybrid
  2. RRF 参数 k：不同值
  3. top-k：不同返回条数
  4. nprobe：dense 检索探测数

算 Recall@k / MRR，输出统一对比表。

用法：
    <venv python> ablation.py
"""
import random
import statistics

import pandas as pd
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
from sentence_transformers import CrossEncoder

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL

# rerank 模型（cross-encoder，专用“相关性判断”，比向量相似度准）
RERANK_MODEL = 'D:/huggingface_cache/models/BAAI--bge-reranker-base/snapshots/master'
RERANK_RECALL_K = 20  # 召回阶段先取 top-20，再精排

CORPUS_PATH = '../data/from_modelscope/corpus-00000-of-00001-a3949861f65a3226.parquet'
QUERIES_PATH = '../data/from_modelscope/queries-00000-of-00001-daeedab899d3c839.parquet'
DEV_PATH = '../data/from_modelscope/dev-00000-of-00001-57fb84a4aceaa695.parquet'

SAMPLE_N = 100  # 抽样 query 数
INSERTED_N = 1000  # 已插入的 corpus 条数


# ---------------- 数据加载（复用，不变） ----------------
def load_valid_queries():
    corpus = pd.read_parquet(CORPUS_PATH)
    dev = pd.read_parquet(DEV_PATH)
    queries = pd.read_parquet(QUERIES_PATH)

    inserted_ids = set(corpus['id'].head(INSERTED_N).tolist())
    qid2pids = dev.groupby('qid')['pid'].apply(list).to_dict()
    valid_qids = [q for q, pids in qid2pids.items() if all(p in inserted_ids for p in pids)]

    random.seed(42)
    sample_qids = random.sample(valid_qids, min(SAMPLE_N, len(valid_qids)))

    qid2text = dict(zip(queries['id'], queries['text']))
    return [{
        'qid': qid,
        'query': qid2text.get(qid, ''),
        'relevant_pids': set(qid2pids[qid]),
    } for qid in sample_qids]


# ---------------- 检索函数（参数化） ----------------
def search_dense(client, query, k, nprobe=16):
    vec = bge_embedding.embed_query(query)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vec],
        anns_field='dense',
        limit=k,
        search_params={'metric_type': 'IP', 'params': {'nprobe': nprobe}},
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


def search_hybrid(client, query, k, rrf_k=100, nprobe=16):
    dense_vec = bge_embedding.embed_query(query)
    req1 = AnnSearchRequest(data=[dense_vec], anns_field='dense',
                            param={'metric_type': 'IP', 'params': {'nprobe': nprobe}}, limit=k)
    req2 = AnnSearchRequest(data=[query], anns_field='sparse',
                            param={'metric_type': 'BM25'}, limit=k)
    res = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[req1, req2],
        ranker=RRFRanker(rrf_k),
        limit=k,
        output_fields=['doc_id'],
    )
    return [hit['entity'].get('doc_id') for hit in res[0]]


def dense_with_text(client, query, recall_k):
    """召回阶段：dense 检索 top-N，同时返回 doc_id 和 text（rerank 需要文本）。"""
    vec = bge_embedding.embed_query(query)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vec],
        anns_field='dense',
        limit=recall_k,
        search_params={'metric_type': 'IP', 'params': {'nprobe': 16}},
        output_fields=['doc_id', 'text'],
    )
    return [(hit['entity'].get('doc_id'), hit['entity'].get('text', '')) for hit in res[0]]


def search_rerank(client, query, k, reranker, recall_k=RERANK_RECALL_K):
    """召回 + 重排：dense 召 top-N，cross-encoder 精排后取 top-k。"""
    candidates = dense_with_text(client, query, recall_k)
    if not candidates:
        return []
    # cross-encoder 对每个 (query, doc) 打分
    pairs = [[query, text] for _, text in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    # 按分数降序排列
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for (doc_id, _), _ in ranked[:k]]


# ---------------- 指标计算 ----------------
import math


def recall_at_k(retrieved, relevant, k):
    """召回率：top-k 里命中的相关文档 / 总相关文档数。"""
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant) if relevant else 0.0


def precision_at_k(retrieved, relevant, k):
    """精确率：top-k 里命中的相关文档 / k。"""
    hit = len(set(retrieved[:k]) & relevant)
    return hit / k if k else 0.0


def mrr(retrieved, relevant):
    """平均倒数排名：第一个相关文档位置的倒数。"""
    for i, pid in enumerate(retrieved):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    """nDCG@k：归一化折损累计增益，衡量排序质量。
    binary relevance（相关=1，不相关=0），增益用 2^rel - 1。
    """
    dcg = 0.0
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant:
            rel = 1.0
            dcg += (2 ** rel - 1) / math.log2(i + 2)  # rank 从1开始，log2(rank+1)
    # 理想排序：所有相关文档都排在最前面
    ideal_hits = min(len(relevant), k)
    idcg = 0.0
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg else 0.0


# ---------------- 通用评估：跑一批 query，算平均指标 ----------------
def evaluate(search_fn, client, queries_data, k):
    """返回 (recall, precision, mrr, ndcg) 四个平均指标。"""
    total_recall = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    n = len(queries_data)
    for item in queries_data:
        retrieved = search_fn(client, item['query'], k)
        total_recall += recall_at_k(retrieved, item['relevant_pids'], k)
        total_precision += precision_at_k(retrieved, item['relevant_pids'], k)
        total_mrr += mrr(retrieved, item['relevant_pids'])
        total_ndcg += ndcg_at_k(retrieved, item['relevant_pids'], k)
    return (
        round(total_recall / n, 4),
        round(total_precision / n, 4),
        round(total_mrr / n, 4),
        round(total_ndcg / n, 4),
    )


# ---------------- 消融实验定义 ----------------
def run_ablation(client, queries_data):
    """依次跑各组消融，返回结果列表。"""
    results = []

    # 实验组 1：检索方式对比（top-k=5 固定）
    print("实验组 1：检索方式对比 (top-k=5)")
    for name, fn in [
        ('dense', lambda c, q, k: search_dense(c, q, k)),
        ('sparse', lambda c, q, k: search_sparse(c, q, k)),
        ('hybrid(rrf=100)', lambda c, q, k: search_hybrid(c, q, k, rrf_k=100)),
    ]:
        rec, prec, mr, ndcg = evaluate(fn, client, queries_data, 5)
        results.append({'ablation': '检索方式', 'config': name, 'k': 5,
                        'recall': rec, 'precision': prec, 'mrr': mr, 'ndcg': ndcg})
        print(f"  {name:<20} recall={rec}  precision={prec}  mrr={mr}  ndcg={ndcg}")

    # 实验组 2：RRF 参数 k 敏感度（top-k=5 固定）
    print("\n实验组 2：RRF 参数敏感度 (top-k=5)")
    for rrf_k in [20, 40, 60, 100]:
        rec, prec, mr, ndcg = evaluate(
            lambda c, q, k: search_hybrid(c, q, k, rrf_k=rrf_k), client, queries_data, 5)
        results.append({'ablation': 'RRF_k', 'config': f'rrf_k={rrf_k}', 'k': 5,
                        'recall': rec, 'precision': prec, 'mrr': mr, 'ndcg': ndcg})
        print(f"  rrf_k={rrf_k:<5} recall={rec}  precision={prec}  mrr={mr}  ndcg={ndcg}")

    # 实验组 3：top-k 敏感度（hybrid rrf=60 固定）
    print("\n实验组 3：top-k 敏感度 (hybrid, rrf=60)")
    for topk in [5, 10, 15, 20, 25]:
        rec, prec, mr, ndcg = evaluate(
            lambda c, q, k: search_hybrid(c, q, k, rrf_k=60), client, queries_data, topk)
        results.append({'ablation': 'top_k', 'config': f'k={topk}', 'k': topk,
                        'recall': rec, 'precision': prec, 'mrr': mr, 'ndcg': ndcg})
        print(f"  top-k={topk:<4} recall={rec}  precision={prec}  mrr={mr}  ndcg={ndcg}")

    # 实验组 4：rerank 消融（有 rerank vs 无 rerank，都用 dense 召回）
    print("\n实验组 4：rerank 消融 (dense 召回)")
    reranker = CrossEncoder(RERANK_MODEL)
    for config_name, fn in [
        ('dense(无rerank)', lambda c, q, k: search_dense(c, q, k)),
        ('dense+rerank', lambda c, q, k: search_rerank(c, q, k, reranker)),
    ]:
        rec, prec, mr, ndcg = evaluate(fn, client, queries_data, 5)
        results.append({'ablation': 'rerank', 'config': config_name, 'k': 5,
                        'recall': rec, 'precision': prec, 'mrr': mr, 'ndcg': ndcg})
        print(f"  {config_name:<16} recall={rec}  precision={prec}  mrr={mr}  ndcg={ndcg}")

    return results


# ---------------- 主流程 ----------------
def main():
    client = MilvusClient(uri=MV_URL)
    queries_data = load_valid_queries()
    print(f"评估 query 数: {len(queries_data)}\n")

    results = run_ablation(client, queries_data)

    # 汇总成 DataFrame 方便看
    print("\n" + "=" * 60)
    print("汇总表")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


def print_qrels():
    corpus = pd.read_parquet(CORPUS_PATH)
    dev = pd.read_parquet(DEV_PATH)
    queries = pd.read_parquet(QUERIES_PATH)

    inserted_ids = set(corpus['id'].head(INSERTED_N).tolist())
    qid2pids = dev.groupby('qid')['pid'].apply(list).to_dict()
    valid_qids = [q for q, pids in qid2pids.items() if all(p in inserted_ids for p in pids)]
    qid2text = dict(zip(queries['id'], queries['text']))
    pid2text = dict(zip(corpus['id'], corpus['text']))

    for q, pids in qid2pids.items():
        if all(p in inserted_ids for p in pids):
            # query 文本
            query_text = qid2text.get(q, '')
            # 多个正样本文档的正文（pids 是列表）
            corpus_texts = [pid2text.get(p, '') for p in pids]

            print(f'query = {query_text}\n')
            for i, ct in enumerate(corpus_texts):
                print(f'corpus[{i}] = {ct}\n')


def analyze_recall(top_k_list=(3, 10)):
    """用正确的 Recall@k 公式统计（命中正样本数 / 总正样本数）。"""
    corpus = pd.read_parquet(CORPUS_PATH)
    dev = pd.read_parquet(DEV_PATH)
    queries = pd.read_parquet(QUERIES_PATH)

    inserted_ids = set(corpus['id'].head(INSERTED_N).tolist())
    qid2pids = dev.groupby('qid')['pid'].apply(list).to_dict()
    qid2text = dict(zip(queries['id'], queries['text']))

    client = MilvusClient(uri=MV_URL)

    valid_qids = [q for q, pids in qid2pids.items() if all(p in inserted_ids for p in pids)]

    # 对每个 query 检索，直接取最大 k（后续按不同 k 截断）
    max_k = max(top_k_list)
    per_query_recall = {k: [] for k in top_k_list}
    per_query_hit = {k: [] for k in top_k_list}

    for q in valid_qids:
        pids = qid2pids[q]
        query_text = qid2text.get(q, '')

        vec = bge_embedding.embed_query(query_text)
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=[vec],
            anns_field='dense',
            limit=max_k,
            search_params={'metric_type': 'IP', 'params': {'nprobe': 16}},
            output_fields=['doc_id'],
        )

        retrieved_ids = [hit.get('entity', hit).get('doc_id') for hit in res[0]]
        pids_set = set(pids)

        for k in top_k_list:
            top = retrieved_ids[:k]
            hit_num = len(set(top) & pids_set)
            recall = hit_num / len(pids)  # 正确 Recall：命中数/总相关数
            hit = 1.0 if hit_num > 0 else 0.0  # Hit：至少1个
            per_query_recall[k].append(recall)
            per_query_hit[k].append(hit)

    # 输出
    print("=" * 60)
    print(f"正确的召回统计（valid query 总数: {len(valid_qids)}）")
    print("=" * 60)
    for k in top_k_list:
        avg_recall = sum(per_query_recall[k]) / len(per_query_recall[k])
        avg_hit = sum(per_query_hit[k]) / len(per_query_hit[k])
        print(f"top-k={k}:  Recall@{k} = {avg_recall:.4f}   Hit@{k} = {avg_hit:.4f}")


def analyze_retrieval(print_details=False):
    TOP_K = 3
    """统计：命中正样本的 query 数，以及每个 query 正样本数的分布。"""
    corpus = pd.read_parquet(CORPUS_PATH)
    dev = pd.read_parquet(DEV_PATH)
    queries = pd.read_parquet(QUERIES_PATH)

    inserted_ids = set(corpus['id'].head(INSERTED_N).tolist())
    qid2pids = dev.groupby('qid')['pid'].apply(list).to_dict()
    qid2text = dict(zip(queries['id'], queries['text']))
    pid2text = dict(zip(corpus['id'], corpus['text']))

    client = MilvusClient(uri=MV_URL)

    valid_qids = [q for q, pids in qid2pids.items() if all(p in inserted_ids for p in pids)]

    hit_count = 0  # 命中了至少一个正样本的 query 数
    total = len(valid_qids)  # 总 valid query 数
    positive_nums = []  # 每个 query 的正样本数

    for q in valid_qids:
        pids = qid2pids[q]
        positive_nums.append(len(pids))

        query_text = qid2text.get(q, '')
        vec = bge_embedding.embed_query(query_text)
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=[vec],
            anns_field='dense',
            limit=TOP_K,
            search_params={'metric_type': 'IP', 'params': {'nprobe': 16}},
            output_fields=['doc_id', 'text'],
        )

        retrieved_ids = set()
        for hit in res[0]:
            e = hit.get('entity', hit)
            retrieved_ids.add(e.get('doc_id'))

        # 命中：检索结果里有至少一个正样本
        is_hit = bool(retrieved_ids & set(pids))
        if is_hit:
            hit_count += 1

        if print_details:
            hit_mark = "✅命中" if is_hit else "❌未命中"
            print(f"[{hit_count}/{total}] {hit_mark} 正样本数={len(pids)} qid={q[:8]}...")
            print(f"  query = {query_text[:50]}")
            print(f"  检索doc_ids = {[i[:8] for i in retrieved_ids]}")
            print(f"  正样本pids = {[p[:8] for p in pids]}")
            print()

    # 统计
    hit_rate = hit_count / total if total else 0
    median_pos = statistics.median(positive_nums) if positive_nums else 0
    avg_pos = sum(positive_nums) / len(positive_nums) if positive_nums else 0

    print("=" * 50)
    print("检索命中统计")
    print("=" * 50)
    print(f"valid query 总数: {total}")
    print(f"命中正样本的 query 数: {hit_count}")
    print(f"命中率 (Recall@{TOP_K}): {hit_rate:.4f}")
    print(
        f"每个 query 正样本数：min={min(positive_nums)}, 中位数={median_pos}, 平均={avg_pos:.2f}, max={max(positive_nums)}")


if __name__ == '__main__':
    # main()
    # print_qrels()
    # print_valid_retrieval()
    # analyze_retrieval()
    analyze_recall()