"""Precision 优化消融实验

目标：在"recall 可接受"的前提下，找出能最大化 precision 的分数阈值方案。

三个消融维度（每个都是单变量）：
  1. dense 检索的 score_threshold（向量相似度阈值）
  2. rerank 后的分数阈值
  3. 两者组合

每个实验算 recall / precision / F1，找 precision-recall 平衡点。

用法：
    <venv python> precision_ablation.py
"""
import random
import math
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL

RERANK_MODEL = 'BAAI/bge-reranker-base'
RECALL_K = 20   # 召回阶段取 top-20

CORPUS_PATH = '../data/from_modelscope/corpus-00000-of-00001-a3949861f65a3226.parquet'
QUERIES_PATH = '../data/from_modelscope/queries-00000-of-00001-daeedab899d3c839.parquet'
DEV_PATH = '../data/from_modelscope/dev-00000-of-00001-57fb84a4aceaa695.parquet'

SAMPLE_N = 100
INSERTED_N = 1000
FINAL_K = 5   # 最终返回条数（评估 precision@5）


# ---------------- 数据加载 ----------------
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


# ---------------- 带分数的检索 ----------------
def dense_search_with_score(client, query, k=RECALL_K, score_threshold=None):
    """dense 检索，返回 [(doc_id, score, text), ...]，可按 score 过滤。"""
    vec = bge_embedding.embed_query(query)
    res = client.search(
        collection_name=COLLECTION_NAME,
        data=[vec],
        anns_field='dense',
        limit=k,
        search_params={'metric_type': 'IP', 'params': {'nprobe': 16}},
        output_fields=['doc_id', 'text'],
    )
    out = []
    for hit in res[0]:
        score = hit.get('distance', 0.0)
        if score_threshold is not None and score < score_threshold:
            continue
        e = hit.get('entity', hit)
        out.append((e.get('doc_id'), score, e.get('text', '')))
    return out


def rerank_with_threshold(candidates, query, reranker, threshold=None):
    """对候选 rerank，按分数降序，可按阈值过滤。"""
    if not candidates:
        return []
    pairs = [[query, text] for _, _, text in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    out = []
    for (doc_id, _, _), s in ranked:
        if threshold is not None and s < threshold:
            continue
        out.append(doc_id)
    return out


# ---------------- 指标 ----------------
def precision_at_k(retrieved, relevant, k):
    """精确率：分母用实际返回数（阈值过滤后可能不足 k）。

    标准 precision@k 用固定 k 做分母；但阈值过滤后实际返回可能 < k，
    此时用固定 k 会低估 precision，所以改为用实际返回数。
    """
    actual = len(retrieved[:k])
    if actual == 0:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / actual


def recall_at_k(retrieved, relevant, k):
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant) if relevant else 0.0


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate(search_fn, client, queries_data, k, reranker=None):
    total_recall = total_precision = 0.0
    n = len(queries_data)
    for item in queries_data:
        retrieved = search_fn(client, item['query'], k, reranker)
        total_recall += recall_at_k(retrieved, item['relevant_pids'], k)
        total_precision += precision_at_k(retrieved, item['relevant_pids'], k)
    rec = total_recall / n
    prec = total_precision / n
    return round(rec, 4), round(prec, 4), round(f1(prec, rec), 4)


# ---------------- 消融实验 ----------------
def run(client, queries_data):
    reranker = CrossEncoder(RERANK_MODEL)
    results = []

    # 实验组 A：dense score_threshold 对照（探查已证明无区分度，仅留无阈值做基准）
    print("实验组 A：dense 阈值对照（探查显示 dense 无区分度，仅保留基准）")
    for th in [None, 0.65]:
        def fn(c, q, k, _rr, th=th):
            cands = dense_search_with_score(c, q, RECALL_K, score_threshold=th)
            return [doc_id for doc_id, _, _ in cands]
        rec, prec, f = evaluate(fn, client, queries_data, FINAL_K, reranker)
        label = "无阈值" if th is None else f"阈值={th}"
        results.append({'组': 'dense阈值', '配置': label, 'recall': rec, 'precision': prec, 'f1': f})
        print(f"  {label:<8} recall={rec}  precision={prec}  f1={f}")

    # 实验组 B：rerank 分数阈值（探查结果显示 rerank 分数在 [0,1]，相关 median≈0.95，无关 median≈0.1）
    print("\n实验组 B：rerank 分数阈值（分数范围 [0,1]，正确阈值区间）")
    for th in [None, 0.3, 0.5, 0.7, 0.9]:
        def fn(c, q, k, rr, th=th):
            cands = dense_search_with_score(c, q, RECALL_K, score_threshold=None)
            return rerank_with_threshold(cands, q, rr, threshold=th)
        rec, prec, f = evaluate(fn, client, queries_data, FINAL_K, reranker)
        label = "无阈值" if th is None else f"阈值={th}"
        results.append({'组': 'rerank阈值', '配置': label, 'recall': rec, 'precision': prec, 'f1': f})
        print(f"  {label:<8} recall={rec}  precision={prec}  f1={f}")

    # 实验组 C：组合（dense 低阈值先粗筛 + rerank 阈值精筛）
    print("\n实验组 C：组合（dense 粗筛 + rerank 精筛）")
    for dense_th, rerank_th in [(0.55, 0.3), (0.55, 0.5), (0.6, 0.5), (0.65, 0.5), (0.55, 0.7)]:
        def fn(c, q, k, rr, dt=dense_th, rt=rerank_th):
            cands = dense_search_with_score(c, q, RECALL_K, score_threshold=dt)
            return rerank_with_threshold(cands, q, rr, threshold=rt)
        rec, prec, f = evaluate(fn, client, queries_data, FINAL_K, reranker)
        label = f"dense>={dense_th}, rerank>={rerank_th}"
        results.append({'组': '组合', '配置': label, 'recall': rec, 'precision': prec, 'f1': f})
        print(f"  {label:<28} recall={rec}  precision={prec}  f1={f}")

    return results


def main():
    client = MilvusClient(uri=MV_URL)
    queries_data = load_valid_queries()
    print(f"评估 query 数: {len(queries_data)}，最终 top-k={FINAL_K}\n")

    results = run(client, queries_data)

    print("\n" + "=" * 70)
    print("汇总（按 f1 降序）")
    print("=" * 70)
    df = pd.DataFrame(results).sort_values('f1', ascending=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
