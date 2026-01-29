import pandas as pd

df = pd.read_csv("ablation_results.csv")

# Overall
overall = df.groupby("System")[["NDCG@10", "Recall@10"]].mean()
print("=== Overall Comparison ===")
print(overall.to_markdown())

# By Complexity
complexity = (
    df.groupby(["System", "Complexity"])[["NDCG@10", "Recall@10"]].mean().unstack()
)
print("\n=== Complexity Breakdown ===")
print(complexity.to_markdown())
