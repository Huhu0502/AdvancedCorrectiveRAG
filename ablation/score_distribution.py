"""分数分布探查脚本 —— 阈值调优前必须先做的一步

回答核心问题：
    相关文档 vs 无关文档，在 dense 分数 和 rerank 分数上，到底有没有区分度？

只有弄清了分数分布，才能判断：
    - 有区分度 → 在两个分布的"缝隙"设阈值
    - 无区分度 → 别设阈值，换更强的 embedding / reranker

用法：
    <venv python> score_distribution.py
"""
import random
import statistics
import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import CrossEncoder

from model.embedding_models import bge_embedding
from utils.env_utils import COLLECTION_NAME, MV_URL

RERANK_MODEL = 'D:/huggingface_cache/models/BAAI--bge-reranker-base/snapshots/master'
RECALL_K = 20

CORPUS_PATH = '../data/from_modelscope/corpus-00000-of-00001-a3949861f65a3226.parquet'
QUERIES_PATH = '../data/from_modelscope/queries-00000-of-00001-daeedab899d3c839.parquet'
DEV_PATH = '../data/from_modelscope/dev-00000-of-00001-57fb84a4aceaa695.parquet'

SAMPLE_N = 30   # 探查用 query 数（不参与最终评估）


def load_queries():
    corpus = pd.read_parquet(CORPUS_PATH)
    dev = pd.read_parquet(DEV_PATH)
    queries = pd.read_parquet(QUERIES_PATH)

    inserted_ids = set(corpus['id'].head(1000).tolist())
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


def collect_scores(client, reranker, queries_data):
    """返回 (rel_dense, rel_rerank, irrel_dense, irrel_rerank) 四个分数列表。"""
    rel_dense, rel_rerank = [], []
    irrel_dense, irrel_rerank = [], []

    for item in queries_data:
        query = item['query']
        relevant = item['relevant_pids']

        # dense 召回 top-20 带分数
        vec = bge_embedding.embed_query(query)
        res = client.search(
            collection_name=COLLECTION_NAME,
            data=[vec],
            anns_field='dense',
            limit=RECALL_K,
            search_params={'metric_type': 'IP', 'params': {'nprobe': 16}},
            output_fields=['doc_id', 'text'],
        )
        candidates = []
        for hit in res[0]:
            e = hit.get('entity', hit)
            candidates.append((e.get('doc_id'), hit.get('distance', 0.0), e.get('text', '')))

        # rerank 打分
        pairs = [[query, text] for _, _, text in candidates]
        rr_scores = reranker.predict(pairs, show_progress_bar=False)

        # 按相关/无关分成两类
        for (doc_id, d_score, _), r_score in zip(candidates, rr_scores):
            if doc_id in relevant:
                rel_dense.append(d_score)
                rel_rerank.append(r_score)
            else:
                irrel_dense.append(d_score)
                irrel_rerank.append(r_score)

    return rel_dense, rel_rerank, irrel_dense, irrel_rerank


def summarize(name, scores):
    if not scores:
        return f"{name}: (无样本)"
    arr = np.array(scores)
    return (
        f"{name}: n={len(arr)}, "
        f"min={arr.min():.3f}, p25={np.percentile(arr,25):.3f}, "
        f"median={np.median(arr):.3f}, p75={np.percentile(arr,75):.3f}, "
        f"max={arr.max():.3f}"
    )


def main():
    client = MilvusClient(uri=MV_URL)
    reranker = CrossEncoder(RERANK_MODEL)
    queries_data = load_queries()
    print(f"探查 query 数: {len(queries_data)}，each 召回 top-{RECALL_K}\n")

    rel_d, rel_r, irrel_d, irrel_r = collect_scores(client, reranker, queries_data)

    print("=" * 70)
    print("Dense 分数（向量内积）")
    print("=" * 70)
    print(summarize("相关文档 dense", rel_d))
    print(summarize("无关文档 dense", irrel_d))

    print("\n" + "=" * 70)
    print("Rerank 分数（cross-encoder logits）")
    print("=" * 70)
    print(summarize("相关文档 rerank", rel_r))
    print(summarize("无关文档 rerank", irrel_r))

    # 区分度判断
    print("\n" + "=" * 70)
    print("区分度判断")
    print("=" * 70)
    if rel_d and irrel_d:
        overlap_dense = min(np.median(rel_d), np.median(irrel_d)) if False else None
        gap_dense = abs(np.median(rel_d) - np.median(irrel_d))
        print(f"dense 中位数差距: {gap_dense:.3f}  "
              f"(相关 median={np.median(rel_d):.3f} vs 无关 median={np.median(irrel_d):.3f})")
    if rel_r and irrel_r:
        gap_rerank = abs(np.median(rel_r) - np.median(irrel_r))
        print(f"rerank 中位数差距: {gap_rerank:.3f}  "
              f"(相关 median={np.median(rel_r):.3f} vs 无关 median={np.median(irrel_r):.3f})")

    print("\n提示：")
    print("  - 若「相关」和「无关」的 median 差距明显（尤其 rerank），则可在缝隙设阈值")
    print("  - 若两者分数高度重叠，说明该分数本身无区分度，应换更强的 embedding / reranker")


if __name__ == '__main__':
    main()
