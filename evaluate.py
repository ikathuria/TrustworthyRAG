import os
import json
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Any
from tqdm import tqdm
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.evaluators.retrieval_evaluator import run_retrieval_evaluation
from src.evaluators.adversarial_evaluator import run_adversarial_evaluation
from src.evaluators.ablation_study import run_ablation_study
from src.evaluators.evaluate_efficiency import run_efficiency_evaluation
from src.evaluators.evaluate_sensitivity import run_sensitivity_analysis
from src.evaluators.ragas_eval import main as run_ragas_evaluation
from src.evaluators.calculate_significance import calculate_significance
from src.evaluators.generator_evaluator import run_generator_evaluation

if __name__ == "__main__":
    print("--- STARTING EVALUATE.PY ---")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "retrieval",
            "adversarial",
            "ablation",
            "efficiency",
            "sensitivity",
            "ragas",
            "significance",
            "generator",
            "all",
        ],
        default="retrieval",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--doc_bench_dir", default="data/raw/DocBench", help="Path to DocBench data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of directories/samples to evaluate",
    )
    parser.add_argument(
        "--output_dir", default="data/results", help="Directory to save results"
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve"
    )
    args = parser.parse_args()

    if args.mode in ["retrieval", "all"]:
        logger.info("Starting Retrieval Evaluation...")
        run_retrieval_evaluation(
            doc_bench_dir=args.doc_bench_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )

    if args.mode in ["adversarial", "all"]:
        logger.info("Starting Adversarial Evaluation...")
        run_adversarial_evaluation(output_dir=args.output_dir, top_k=args.top_k)

    if args.mode in ["ablation", "all"]:
        logger.info("Starting Ablation Study...")
        run_ablation_study(
            doc_bench_dir=args.doc_bench_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )

    if args.mode in ["efficiency", "all"]:
        logger.info("Starting Efficiency Evaluation...")
        run_efficiency_evaluation(
            doc_bench_dir=args.doc_bench_dir,
            limit=args.limit or 10,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )

    if args.mode in ["sensitivity", "all"]:
        logger.info("Starting Sensitivity Analysis...")
        # Use first DocBench dir as target if not specified
        subdirs = [
            d
            for d in os.listdir(args.doc_bench_dir)
            if os.path.isdir(os.path.join(args.doc_bench_dir, d))
        ]
        if subdirs:
            target_dir = os.path.join(args.doc_bench_dir, subdirs[0])
            betas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            run_sensitivity_analysis(
                target_dir=target_dir,
                betas=betas,
                output_dir=args.output_dir,
                top_k=args.top_k,
            )

    if args.mode in ["ragas", "all"]:
        logger.info("Starting Ragas Evaluation Prep...")
        run_ragas_evaluation(
            doc_bench_dir=args.doc_bench_dir,
            num_samples=args.limit or 5,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )

    if args.mode in ["significance", "all"]:
        logger.info("Calculating Statistical Significance...")
        eval_csv = os.path.join(args.output_dir, "evaluation_results.csv")
        if os.path.exists(eval_csv):
            calculate_significance(eval_csv)
        else:
            logger.warning(f"Evaluation results not found at {eval_csv}")

    if args.mode in ["generator", "all"]:
        logger.info("Starting Generator Evaluation Accuracy...")
        run_generator_evaluation(
            doc_bench_dir=args.doc_bench_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
