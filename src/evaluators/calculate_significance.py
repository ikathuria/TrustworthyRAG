import pandas as pd
from scipy import stats
import numpy as np


def calculate_significance(csv_path):
    df = pd.read_csv(csv_path)

    # Extract QALF and Vector-Only NDCG scores
    qalf_ndcg = df[df["System"] == "qalf"]["NDCG@10"].values
    vector_ndcg = df[df["System"] == "vector_only"]["NDCG@10"].values

    # Ensure they have the same number of samples (DocBench queries)
    # The evaluation file might have different counts if some failed, so we align by Query_ID
    qalf_df = df[df["System"] == "qalf"][["Query_ID", "NDCG@10"]].set_index("Query_ID")
    vector_df = df[df["System"] == "vector_only"][["Query_ID", "NDCG@10"]].set_index(
        "Query_ID"
    )

    # Common indices
    common_idx = qalf_df.index.intersection(vector_df.index)

    q_scores = qalf_df.loc[common_idx, "NDCG@10"].values
    v_scores = vector_df.loc[common_idx, "NDCG@10"].values

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(q_scores, v_scores)

    print(f"--- Statistical Significance Results ---")
    print(f"Samples: {len(common_idx)}")
    print(f"Mean NDCG (QALF): {np.mean(q_scores):.4f}")
    print(f"Mean NDCG (Vector): {np.mean(v_scores):.4f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4e}")

    if p_value < 0.05:
        print("Result: Statistically significant (p < 0.05)")
    else:
        print("Result: Not statistically significant")


if __name__ == "__main__":
    import os

    default_path = os.path.join("data", "results", "evaluation_results.csv")
    calculate_significance(default_path)
