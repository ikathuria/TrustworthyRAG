import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import logging
from langchain_ollama import OllamaLLM
from src.utils.systems import SystemRegistry
from src.neo4j.neo4j_manager import Neo4jManager
import src.utils.constants as C

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_PROMPT = """Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {question}
User Answer: {sys_ans}
Reference Answer: {ref_ans}
Reference Text: {ref_text}

Evaluation Form (score ONLY):
- Correctness: """


def run_generator_evaluation(
    doc_bench_dir: str,
    limit: int = None,
    output_dir: str = "data/results",
    top_k: int = 10,
):
    logger.info(f"Initializing generator evaluation with top_k={top_k}...")

    neo4j_manager = Neo4jManager(
        uri=C.NEO4J_URI, username=C.NEO4J_USERNAME, password=C.NEO4J_PASSWORD
    )

    registry = SystemRegistry(neo4j_manager=neo4j_manager)
    llm = OllamaLLM(model=C.GENERATOR_MODEL, temperature=0, base_url=C.OLLAMA_URI)

    # Domain mapping
    # Academia: 0-48, Finance: 49-88, Government: 89-132, Law: 133-178, News: 179-228
    domains = {
        "Academia": range(0, 49),
        "Finance": range(49, 89),
        "Government": range(89, 133),
        "Law": range(133, 179),
        "News": range(179, 229),
    }

    output_path = os.path.join(output_dir, "generator_evaluation.csv")
    existing_keys = set()

    # Load existing results for resume
    if os.path.exists(output_path):
        try:
            temp_df = pd.read_csv(output_path)
            for _, row in temp_df.iterrows():
                # Store (Subdir, Question) as key for skipping
                existing_keys.add((str(row["Subdir"]), str(row["Question"])))
            logger.info(
                f"Loaded {len(existing_keys)} existing results from {output_path}. Resuming..."
            )
        except Exception as e:
            logger.warning(f"Could not load existing results: {e}. Starting fresh.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def get_domain(idx):
        for domain, r in domains.items():
            if idx in r:
                return domain
        return "Unknown"

    # Get all subdirectories and sort numerically
    subdirs = [
        d
        for d in os.listdir(doc_bench_dir)
        if os.path.isdir(os.path.join(doc_bench_dir, d))
    ]
    subdirs.sort(key=lambda x: int(x) if x.isdigit() else 999)

    if limit:
        subdirs = subdirs[:limit]

    for subdir in tqdm(subdirs, desc="Evaluating Domains"):
        subdir_idx = int(subdir) if subdir.isdigit() else -1
        domain = get_domain(subdir_idx)

        qa_file = os.path.join(doc_bench_dir, subdir, f"{subdir}_qa.jsonl")
        if not os.path.exists(qa_file):
            continue

        with open(qa_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                qa_data = json.loads(line)
                question = qa_data["question"]

                # Check if already scored in existing_keys (Skip expensive calls)
                if (str(subdir), str(question)) in existing_keys:
                    continue

                ref_ans = qa_data["answer"]
                ref_text = qa_data.get("evidence", "")

                # Run QALF
                try:
                    sys_res = registry.run_qalf(question, top_k=top_k)
                    sys_ans = sys_res["answer"]
                except Exception as e:
                    logger.error(f"Error running QALF for {subdir}/{line_idx}: {e}")
                    sys_ans = ""

                # Score with LLM
                score = 0
                if sys_ans and sys_ans.strip() != "0":
                    prompt = EVAL_PROMPT.format(
                        question=question,
                        sys_ans=sys_ans,
                        ref_ans=ref_ans,
                        ref_text=ref_text,
                    )
                    try:
                        response = llm.invoke(prompt)
                        # Extract 0 or 1 from response
                        if "1" in response:
                            score = 1
                        elif "0" in response:
                            score = 0
                        else:
                            # Heuristic: if response is empty or weird, default to 0
                            score = 0
                    except Exception as e:
                        logger.error(f"Error scoring with LLM: {e}")
                        score = 0

                new_row = {
                    "Subdir": subdir,
                    "Domain": domain,
                    "Question": question,
                    "Ref_Ans": ref_ans,
                    "Sys_Ans": sys_ans,
                    "Score": score,
                }

                # Real-time append to CSV
                row_df = pd.DataFrame([new_row])
                file_exists = os.path.isfile(output_path)
                row_df.to_csv(
                    output_path,
                    mode="a",
                    index=False,
                    header=not file_exists,
                    encoding="utf-8",
                )

    # Final Reload to calculate statistics
    if not os.path.exists(output_path):
        logger.warning("No results generated.")
        return

    df = pd.read_csv(output_path)
    logger.info(f"Evaluation complete. Total results: {len(df)}")

    # Calculate domain averages
    domain_scores = df.groupby("Domain")["Score"].mean() * 100
    overall_score = df["Score"].mean() * 100

    # Create LaTeX Table
    # Ensure all domains are present in order
    ordered_domains = ["Academia", "Finance", "Government", "Law", "News"]
    domain_results = {d: domain_scores.get(d, 0.0) for d in ordered_domains}

    latex_table = f"""\\begin{{table*}}[h]
\\centering
\\begin{{tabular}}{{ccccccc}}
\\hline
\\textbf{{System}} & \\multicolumn{{5}}{{c}}{{\\textbf{{Domains}}}}& \\textbf{{Overall}}\\\\
 &  Academia& Finance& Government& Law& News&\\\\
\\hline
Llama-3 8b& 44.6& 23.6& 61.5& 54.5& 86.6& 49.6\\\\
            MMGR& 60.5& 65.8& 66.5& 70.4& 77.1& 76.8\\\\
RAGAnything& 61.4& 67.0& 61.5& 60.2& 66.3& 63.4\\\\
            \\textbf{{QALF (Ours)}}& {domain_results['Academia']:.1f}& {domain_results['Finance']:.1f}& {domain_results['Government']:.1f}& {domain_results['Law']:.1f}& {domain_results['News']:.1f}& {overall_score:.1f}\\\\
\\hline
\\end{{tabular}}
\\caption{{Generator accuracy on DocBench dataset}}
\\label{{tab:complexity_results}}
\\end{{table*}}"""

    print("\n=== GENERATOR EVALUATION RESULTS (LaTeX) ===")
    print(latex_table)
    print("\n")

    neo4j_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_bench_dir", default="data/raw/DocBench")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_generator_evaluation(args.doc_bench_dir, args.limit)
