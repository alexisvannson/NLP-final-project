"""Compare all summarization models on the same test set.

This script runs all 6 models (TextRank, LexRank, BART, Hierarchical, Longformer, Sliding Window)
on the same test samples and generates comprehensive comparison tables and visualizations.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import ComprehensiveEvaluator


def load_test_data(
    dataset_name: str, num_samples: Optional[int] = None
) -> List[Dict]:
    """Load test data from processed datasets.

    Args:
        dataset_name: Name of dataset (arxiv, pubmed, etc.)
        num_samples: Number of samples to load (None = all)

    Returns:
        List of test samples
    """
    data_path = Path("data/processed") / dataset_name / "test.json"

    if not data_path.exists():
        print(f"Warning: Test data not found at {data_path}")
        return []

    with open(data_path, "r") as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    print(f"Loaded {len(data)} samples from {dataset_name}")
    return data


def run_model(
    model_name: str, samples: List[Dict], device: str = "cpu"
) -> Dict[str, List]:
    """Run a single model on test samples.

    Args:
        model_name: Name of model to run
        samples: Test samples
        device: Device to use

    Returns:
        Dictionary with predictions, sources, references, and timing
    """
    from models.baseline_abstractive import BARTChunkSummarizer
    from models.baseline_extractive import LexRankSummarizer, TextRankSummarizer
    from models.hierarchical_transformer import HierarchicalTransformerSummarizer
    from models.longformer_summarizer import LongformerSummarizer
    from models.sliding_window import SlidingWindowSummarizer

    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()}")
    print(f"{'='*60}")

    # Initialize model
    model = None
    try:
        if model_name == "textrank":
            model = TextRankSummarizer(num_sentences=5)
        elif model_name == "lexrank":
            model = LexRankSummarizer(num_sentences=5)
        elif model_name == "bart":
            model = BARTChunkSummarizer(device=device)
        elif model_name == "hierarchical":
            model = HierarchicalTransformerSummarizer(device=device)
        elif model_name == "longformer":
            model = LongformerSummarizer(device=device)
        elif model_name == "sliding_window":
            model = SlidingWindowSummarizer(device=device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        print(f"Error initializing {model_name}: {e}")
        print(f"Skipping {model_name}")
        return None

    # Generate predictions
    predictions = []
    sources = []
    references = []
    times = []

    for sample in tqdm(samples, desc=f"{model_name}"):
        source = sample.get("article", sample.get("document", ""))
        reference = sample.get("summary", sample.get("abstract", ""))

        sources.append(source)
        references.append(reference)

        try:
            start_time = time.time()
            prediction = model.summarize(source)
            elapsed = time.time() - start_time

            predictions.append(prediction)
            times.append(elapsed)
        except Exception as e:
            print(f"Error generating summary: {e}")
            predictions.append("")
            times.append(0)

    avg_time = np.mean(times) if times else 0
    print(f"Average time per sample: {avg_time:.2f}s")

    return {
        "predictions": predictions,
        "sources": sources,
        "references": references,
        "times": times,
        "avg_time": avg_time,
    }


def evaluate_model(
    model_name: str, results: Dict, metrics: List[str]
) -> Dict:
    """Evaluate model predictions.

    Args:
        model_name: Name of model
        results: Results from run_model
        metrics: List of metrics to compute

    Returns:
        Dictionary of evaluation metrics
    """
    if results is None:
        return None

    print(f"\nEvaluating {model_name}...")

    evaluator = ComprehensiveEvaluator(metrics=metrics)

    scores = evaluator.evaluate(
        predictions=results["predictions"],
        references=results["references"],
        sources=results["sources"],
    )

    # Add timing info
    scores["avg_time"] = results["avg_time"]

    return scores


def create_comparison_table(all_scores: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from all model scores.

    Args:
        all_scores: Dictionary mapping model names to their scores

    Returns:
        DataFrame with comparison
    """
    rows = []

    for model_name, scores in all_scores.items():
        if scores is None:
            continue

        row = {"Model": model_name}
        row.update(scores)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns for better readability
    priority_cols = [
        "Model",
        "rouge1_fmeasure",
        "rouge2_fmeasure",
        "rougeL_fmeasure",
        "bertscore_f1",
        "faithfulness_mean",
        "coverage_mean",
        "redundancy_mean",
        "avg_time",
    ]

    cols = [c for c in priority_cols if c in df.columns]
    cols += [c for c in df.columns if c not in cols]

    return df[cols]


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations.

    Args:
        df: Comparison dataframe
        output_dir: Directory to save plots
    """
    sns.set_style("whitegrid")

    # Plot 1: ROUGE scores comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rouge_metrics = ["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]
    rouge_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    for idx, (metric, label) in enumerate(zip(rouge_metrics, rouge_labels)):
        if metric in df.columns:
            ax = axes[idx]
            df_sorted = df.sort_values(metric, ascending=True)
            ax.barh(df_sorted["Model"], df_sorted[metric])
            ax.set_xlabel("F1 Score")
            ax.set_title(label)
            ax.set_xlim(0, 1)

            # Add value labels
            for i, v in enumerate(df_sorted[metric]):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "rouge_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'rouge_comparison.png'}")
    plt.close()

    # Plot 2: Quality metrics (BERTScore, Faithfulness, Coverage)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    quality_metrics = [
        ("bertscore_f1", "BERTScore F1"),
        ("faithfulness_mean", "Faithfulness"),
        ("coverage_mean", "Coverage"),
    ]

    for idx, (metric, label) in enumerate(quality_metrics):
        if metric in df.columns:
            ax = axes[idx]
            df_sorted = df.sort_values(metric, ascending=True)
            ax.barh(df_sorted["Model"], df_sorted[metric])
            ax.set_xlabel("Score")
            ax.set_title(label)
            ax.set_xlim(0, 1)

            # Add value labels
            for i, v in enumerate(df_sorted[metric]):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    plt.savefig(output_dir / "quality_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'quality_comparison.png'}")
    plt.close()

    # Plot 3: Speed vs Quality tradeoff
    if "avg_time" in df.columns and "rouge1_fmeasure" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(
            df["avg_time"],
            df["rouge1_fmeasure"],
            s=200,
            alpha=0.6,
            c=range(len(df)),
            cmap="viridis",
        )

        for idx, row in df.iterrows():
            ax.annotate(
                row["Model"],
                (row["avg_time"], row["rouge1_fmeasure"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.set_xlabel("Average Time per Sample (seconds)")
        ax.set_ylabel("ROUGE-1 F1 Score")
        ax.set_title("Speed vs Quality Tradeoff")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "speed_vs_quality.png", dpi=300, bbox_inches="tight"
        )
        print(f"Saved: {output_dir / 'speed_vs_quality.png'}")
        plt.close()

    # Plot 4: Radar chart for overall comparison
    try:
        from math import pi

        categories = ["ROUGE-1", "ROUGE-L", "BERTScore", "Faithfulness", "Coverage"]
        metric_keys = [
            "rouge1_fmeasure",
            "rougeL_fmeasure",
            "bertscore_f1",
            "faithfulness_mean",
            "coverage_mean",
        ]

        # Check if all metrics are available
        if all(k in df.columns for k in metric_keys):
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]

            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)

            for idx, row in df.iterrows():
                values = [row[k] for k in metric_keys]
                values += values[:1]

                ax.plot(angles, values, "o-", linewidth=2, label=row["Model"])
                ax.fill(angles, values, alpha=0.15)

            ax.set_ylim(0, 1)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.set_title("Overall Model Comparison", y=1.08, fontsize=14)

            plt.tight_layout()
            plt.savefig(
                output_dir / "radar_comparison.png", dpi=300, bbox_inches="tight"
            )
            print(f"Saved: {output_dir / 'radar_comparison.png'}")
            plt.close()
    except Exception as e:
        print(f"Could not create radar chart: {e}")


def generate_report(df: pd.DataFrame, output_dir: Path, dataset_name: str, num_samples: int):
    """Generate markdown report.

    Args:
        df: Comparison dataframe
        output_dir: Directory to save report
        dataset_name: Name of dataset
        num_samples: Number of samples evaluated
    """
    report_path = output_dir / "comparison_report.md"

    with open(report_path, "w") as f:
        f.write(f"# Model Comparison Report\n\n")
        f.write(f"**Dataset:** {dataset_name}\n")
        f.write(f"**Samples:** {num_samples}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Results Summary\n\n")

        # Format table for markdown
        f.write("### ROUGE Scores\n\n")
        rouge_cols = ["Model", "rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure"]
        rouge_cols = [c for c in rouge_cols if c in df.columns]
        if rouge_cols:
            rouge_df = df[rouge_cols].copy()
            rouge_df.columns = ["Model", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
            f.write(rouge_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

        f.write("### Quality Metrics\n\n")
        quality_cols = [
            "Model",
            "bertscore_f1",
            "faithfulness_mean",
            "coverage_mean",
            "redundancy_mean",
        ]
        quality_cols = [c for c in quality_cols if c in df.columns]
        if quality_cols:
            quality_df = df[quality_cols].copy()
            quality_df.columns = ["Model", "BERTScore", "Faithfulness", "Coverage", "Redundancy"]
            f.write(quality_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

        f.write("### Performance\n\n")
        if "avg_time" in df.columns:
            perf_df = df[["Model", "avg_time"]].copy()
            perf_df.columns = ["Model", "Avg Time (s)"]
            f.write(perf_df.to_markdown(index=False, floatfmt=".2f"))
            f.write("\n\n")

        f.write("## Key Findings\n\n")

        # Identify best models
        if "rouge1_fmeasure" in df.columns:
            best_rouge = df.loc[df["rouge1_fmeasure"].idxmax()]
            f.write(f"- **Best ROUGE-1:** {best_rouge['Model']} ({best_rouge['rouge1_fmeasure']:.4f})\n")

        if "bertscore_f1" in df.columns:
            best_bert = df.loc[df["bertscore_f1"].idxmax()]
            f.write(f"- **Best BERTScore:** {best_bert['Model']} ({best_bert['bertscore_f1']:.4f})\n")

        if "faithfulness_mean" in df.columns:
            best_faith = df.loc[df["faithfulness_mean"].idxmax()]
            f.write(
                f"- **Most Faithful:** {best_faith['Model']} ({best_faith['faithfulness_mean']:.4f})\n"
            )

        if "avg_time" in df.columns:
            fastest = df.loc[df["avg_time"].idxmin()]
            f.write(f"- **Fastest:** {fastest['Model']} ({fastest['avg_time']:.2f}s)\n")

        f.write("\n## Approach Comparison\n\n")
        f.write("### Hierarchical vs Sliding Window vs Sparse Attention\n\n")

        approaches = {
            "Hierarchical": ["hierarchical"],
            "Sliding Window": ["sliding_window", "bart"],
            "Sparse Attention": ["longformer"],
        }

        for approach_name, model_list in approaches.items():
            approach_models = df[df["Model"].isin(model_list)]
            if not approach_models.empty:
                f.write(f"**{approach_name}:**\n")
                for _, row in approach_models.iterrows():
                    if "rouge1_fmeasure" in row:
                        f.write(
                            f"- {row['Model']}: ROUGE-1={row['rouge1_fmeasure']:.4f}"
                        )
                        if "avg_time" in row:
                            f.write(f", Time={row['avg_time']:.2f}s")
                        f.write("\n")
                f.write("\n")

        f.write("## Visualizations\n\n")
        f.write("![ROUGE Comparison](rouge_comparison.png)\n\n")
        f.write("![Quality Comparison](quality_comparison.png)\n\n")
        f.write("![Speed vs Quality](speed_vs_quality.png)\n\n")
        f.write("![Radar Comparison](radar_comparison.png)\n\n")

    print(f"\nReport saved to: {report_path}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare all summarization models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="arxiv",
        help="Dataset to use (arxiv, pubmed, booksum, billsum)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["textrank", "lexrank", "bart", "hierarchical", "longformer", "sliding_window"],
        help="Models to compare",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["rouge", "bertscore", "faithfulness", "coverage", "redundancy"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load test data
    samples = load_test_data(args.dataset, args.num_samples)

    if not samples:
        print(
            "\nERROR: No test data found. Please run 'make preprocess' first to prepare datasets."
        )
        sys.exit(1)

    # Run all models
    all_results = {}
    all_scores = {}

    for model_name in args.models:
        results = run_model(model_name, samples, args.device)
        if results:
            all_results[model_name] = results

            # Save predictions
            pred_file = output_dir / f"{model_name}_predictions.json"
            with open(pred_file, "w") as f:
                json.dump(
                    {
                        "predictions": results["predictions"],
                        "references": results["references"],
                        "sources": results["sources"],
                    },
                    f,
                    indent=2,
                )
            print(f"Saved predictions to {pred_file}")

            # Evaluate
            scores = evaluate_model(model_name, results, args.metrics)
            if scores:
                all_scores[model_name] = scores

    # Create comparison table
    if all_scores:
        print(f"\n{'='*60}")
        print("CREATING COMPARISON TABLE")
        print(f"{'='*60}")

        df = create_comparison_table(all_scores)

        # Save to CSV
        csv_file = output_dir / "comparison_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSaved results to: {csv_file}")

        # Print to console
        print("\n" + df.to_string(index=False))

        # Create visualizations
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        plot_comparison(df, output_dir)

        # Generate report
        print(f"\n{'='*60}")
        print("GENERATING REPORT")
        print(f"{'='*60}")
        generate_report(df, output_dir, args.dataset, len(samples))

        # Save scores
        scores_file = output_dir / "all_scores.json"
        with open(scores_file, "w") as f:
            json.dump(all_scores, f, indent=2)
        print(f"\nAll scores saved to: {scores_file}")

        print(f"\n{'='*60}")
        print("COMPARISON COMPLETE!")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_dir}")
        print(f"View report: {output_dir / 'comparison_report.md'}")
        print(f"{'='*60}\n")
    else:
        print("\nERROR: No models successfully evaluated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
