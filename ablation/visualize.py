"""消融实验结果可视化 —— 按分组分别画图

每个 ablation 组画一张图，把四个指标（recall/precision/mrr/ndcg）并列对比。

用法：
    <venv python> visualize.py
"""
import matplotlib.pyplot as plt
import matplotlib


def setup_chinese_font():
    for font in ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Noto Sans CJK SC', 'DejaVu Sans']:
        try:
            matplotlib.rcParams['font.sans-serif'] = [font]
            matplotlib.rcParams['axes.unicode_minus'] = False
            return
        except Exception:
            continue
    matplotlib.rcParams['axes.unicode_minus'] = False


# ============ 数据：本次消融实验结果 ============
# 每条：dict(group=分组名, config=配置名, recall, precision, mrr, ndcg)
RESULTS = [
    # 实验组1：检索方式
    {"group": "检索方式", "config": "dense",      "recall": 0.6613, "precision": 0.2040, "mrr": 0.6180, "ndcg": 0.5832},
    {"group": "检索方式", "config": "sparse",     "recall": 0.3600, "precision": 0.1140, "mrr": 0.3292, "ndcg": 0.3059},
    {"group": "检索方式", "config": "hybrid(RRF)","recall": 0.5897, "precision": 0.1820, "mrr": 0.5415, "ndcg": 0.5094},

    # 实验组2：RRF 参数
    {"group": "RRF_k", "config": "rrf_k=20",  "recall": 0.5897, "precision": 0.1820, "mrr": 0.5415, "ndcg": 0.5094},
    {"group": "RRF_k", "config": "rrf_k=40",  "recall": 0.5897, "precision": 0.1820, "mrr": 0.5415, "ndcg": 0.5094},
    {"group": "RRF_k", "config": "rrf_k=60",  "recall": 0.5897, "precision": 0.1820, "mrr": 0.5415, "ndcg": 0.5094},
    {"group": "RRF_k", "config": "rrf_k=100", "recall": 0.5897, "precision": 0.1820, "mrr": 0.5415, "ndcg": 0.5094},

    # 实验组3：top-k
    {"group": "top_k", "config": "k=5",  "recall": 0.5897, "precision": 0.1820, "mrr": 0.5415, "ndcg": 0.5094},
    {"group": "top_k", "config": "k=10", "recall": 0.7134, "precision": 0.1120, "mrr": 0.5690, "ndcg": 0.5545},
    {"group": "top_k", "config": "k=15", "recall": 0.7601, "precision": 0.0807, "mrr": 0.5633, "ndcg": 0.5654},
    {"group": "top_k", "config": "k=20", "recall": 0.8288, "precision": 0.0670, "mrr": 0.5551, "ndcg": 0.5800},
    {"group": "top_k", "config": "k=25", "recall": 0.8451, "precision": 0.0548, "mrr": 0.5480, "ndcg": 0.5773},

    # 实验组4：rerank
    {"group": "rerank", "config": "dense(无rerank)", "recall": 0.6613, "precision": 0.2040, "mrr": 0.6180, "ndcg": 0.5832},
    {"group": "rerank", "config": "dense+rerank",    "recall": 0.6710, "precision": 0.2160, "mrr": 0.6520, "ndcg": 0.6167},
]


METRICS = [
    ("recall", "steelblue"),
    ("precision", "darkorange"),
    ("mrr", "seagreen"),
    ("ndcg", "crimson"),
]


def plot_group_bar(group, rows, out_path):
    """一个分组画一张图：x 轴是各配置，四组柱（四个指标）并列。"""
    configs = [r["config"] for r in rows]
    x = range(len(configs))

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 2.2), 6))
    width = 0.18
    offset = [-1.5, -0.5, 0.5, 1.5]   # 四组柱的相对偏移

    for (metric_name, color), off in zip(METRICS, offset):
        values = [r[metric_name] for r in rows]
        bars = ax.bar([i + off * width for i in x], values, width=width,
                      color=color, alpha=0.85, label=metric_name.upper())
        # 数值标注
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=40)

    ax.set_xticks(list(x))
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title(f"消融实验：{group}", fontsize=14)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")
    plt.close(fig)


def plot_topk_line(rows, out_path="topk_line.png"):
    """top-k 组单独画折线图：四个指标随 k 变化。"""
    rows_sorted = sorted(rows, key=lambda r: int(r["config"].split("=")[1]))
    ks = [int(r["config"].split("=")[1]) for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric_name, color in METRICS:
        values = [r[metric_name] for r in rows_sorted]
        ax.plot(ks, values, marker="o", color=color, label=metric_name.upper())

    ax.set_xlabel("top-k")
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 1.0)
    ax.set_title("指标随 top-k 变化")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")
    plt.close(fig)


def main():
    setup_chinese_font()

    # 按分组聚合
    from collections import defaultdict, OrderedDict
    groups = OrderedDict()
    for r in RESULTS:
        groups.setdefault(r["group"], []).append(r)

    # 每个分组画一张柱状图（四指标并列）
    for i, (group, rows) in enumerate(groups.items(), 1):
        plot_group_bar(group, rows, f"ablation_{i}_{group}.png")

    # top-k 额外画折线图
    if "top_k" in groups:
        plot_topk_line(groups["top_k"])

    print("\n全部图已生成。")


if __name__ == "__main__":
    main()
