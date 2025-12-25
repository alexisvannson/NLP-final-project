"""Analyze performance degradation with document length.

This script evaluates how model performance changes as document length increases,
helping identify which approaches handle longer documents better.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import ComprehensiveEvaluator


def count_tokens(text: str) -> int:
    """Count tokens in text (simple word-based approximation).

    Args:
        text: Input text

    Returns:
        Token count
    """
    return len(text.split())


def bin_by_length(
    samples: List[Dict], bin_ranges: List[Tuple[int, int]]
) -> Dict[str, List[Dict]]:
    """Bin samples by document length.

    Args:
        samples: List of samples with 'article'/'document' and 'summary' keys
        bin_ranges: List of (min_length, max_length) tuples

    Returns:
        Dictionary mapping bin names to samples
    """
    bins = defaultdict(list)

    for sample in samples:
        source = sample.get("article", sample.get("document", ""))
        length = count_tokens(source)

        # Find appropriate bin
        for min_len, max_len in bin_ranges:
            if min_len <= length < max_len:
                bin_name = f"{min_len//1000}K-{max_len//1000}K"
                bins[bin_name].append(sample)
                break

    return bins


def run_model_on_bin(
    model_name: str,
    samples: List[Dict],
    device: str = "cpu"
) -> Dict:
    """Run model on samples from a specific length bin.

    Args:
        model_name: Name of model
        samples: Samples in this bin
        device: Device to use

    Returns:
        Dictionary with predictions, sources, references
    """
    from models.baseline_abstractive import BARTChunkSummarizer
    from models.baseline_extractive import LexRankSummarizer, TextRankSummarizer
    from models.hierarchical_transformer import HierarchicalTransformerSummarizer
    from models.longformer_summarizer import LongformerSummarizer
    from models.sliding_window import SlidingWindowSummarizer

    # Initialize model
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
        return None

    # Generate predictions
    predictions = []
    sources = []
    references = []

    for sample in tqdm(samples, desc=f"{model_name}", leave=False):
        source = sample.get("article", sample.get("document", ""))
        reference = sample.get("summary", sample.get("abstract", ""))

        sources.append(source)
        references.append(reference)

        try:
            prediction = model.summarize(source)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error: {e}")
            predictions.append("")

    return {
        "predictions": predictions,
        "sources": sources,
        "references": references,
    }


def analyze_model_by_length(
    model_name: str,
    binned_samples: Dict[str, List[Dict]],
    metrics: List[str],
    device: str = "cpu"
) -> Dict[str, Dict]:
    """Analyze a single model across different length bins.

    Args:
        model_name: Name of model
        binned_samples: Dictionary of binned samples
        metrics: Metrics to compute
        device: Device to use

    Returns:
        Dictionary mapping bin names to metric scores
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name.upper()} by length")
    print(f"{'='*60}")

    results = {}

    for bin_name in sorted(binned_samples.keys()):
        samples = binned_samples[bin_name]

        if not samples:
            continue

        print(f"\nBin: {bin_name} ({len(samples)} samples)")

        # Run model
        model_results = run_model_on_bin(model_name, samples, device)

        if model_results is None:
            continue

        # Evaluate
        evaluator = ComprehensiveEvaluator(metrics=metrics)
        scores = evaluator.evaluate(
            predictions=model_results["predictions"],
            references=model_results["references"],
            sources=model_results["sources"],
        )

        results[bin_name] = scores

        # Print key metrics
        if "rouge1_fmeasure" in scores:
            print(f"  ROUGE-1: {scores['rouge1_fmeasure']:.4f}")
        if "bertscore_f1" in scores:
            print(f"  BERTScore: {scores['bertscore_f1']:.4f}")

    return results


def compute_degradation_rate(results: Dict[str, Dict], metric_key: str) -> float:
    """Compute degradation rate for a metric across bins.

    Args:
        results: Results by bin
        metric_key: Metric to analyze

    Returns:
        Degradation rate (negative slope)
    """
    if not results:
        return 0.0

    # Extract scores in order
    bins = sorted(results.keys())
    scores = [results[b].get(metric_key, 0) for b in bins]

    if len(scores) < 2:
        return 0.0

    # Linear regression to get slope
    x = np.arange(len(scores))
    slope, _, _, _, _ = stats.linregress(x, scores)

    return slope


def plot_length_analysis(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: Path
):
    """Create visualizations for length analysis.

    Args:
        all_results: Results for all models and bins
        output_dir: Directory to save plots
    """
    sns.set_style("whitegrid")

    # Get bins in order
    sample_results = next(iter(all_results.values()))
    bins = sorted(sample_results.keys())
    bin_indices = np.arange(len(bins))

    # Plot 1: ROUGE-1 vs Document Length
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, results in all_results.items():
        scores = [results.get(b, {}).get("rouge1_fmeasure", 0) for b in bins]
        ax.plot(bin_indices, scores, marker='o', linewidth=2, label=model_name, markersize=8)

    ax.set_xticks(bin_indices)
    ax.set_xticklabels(bins, rotation=45, ha='right')
    ax.set_xlabel("Document Length", fontsize=12)
    ax.set_ylabel("ROUGE-1 F1 Score", fontsize=12)
    ax.set_title("Performance Degradation: ROUGE-1 vs Document Length", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "rouge1_vs_length.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'rouge1_vs_length.png'}")
    plt.close()

    # Plot 2: BERTScore vs Document Length
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, results in all_results.items():
        scores = [results.get(b, {}).get("bertscore_f1", 0) for b in bins]
        if any(scores):  # Only plot if we have BERTScore data
            ax.plot(bin_indices, scores, marker='s', linewidth=2, label=model_name, markersize=8)

    ax.set_xticks(bin_indices)
    ax.set_xticklabels(bins, rotation=45, ha='right')
    ax.set_xlabel("Document Length", fontsize=12)
    ax.set_ylabel("BERTScore F1", fontsize=12)
    ax.set_title("Semantic Similarity vs Document Length", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "bertscore_vs_length.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'bertscore_vs_length.png'}")
    plt.close()

    # Plot 3: Faithfulness vs Document Length
    fig, ax = plt.subplots(figsize=(12, 6))

    has_faithfulness = False
    for model_name, results in all_results.items():
        scores = [results.get(b, {}).get("faithfulness_mean", 0) for b in bins]
        if any(scores):
            has_faithfulness = True
            ax.plot(bin_indices, scores, marker='^', linewidth=2, label=model_name, markersize=8)

    if has_faithfulness:
        ax.set_xticks(bin_indices)
        ax.set_xticklabels(bins, rotation=45, ha='right')
        ax.set_xlabel("Document Length", fontsize=12)
        ax.set_ylabel("Faithfulness Score", fontsize=12)
        ax.set_title("Faithfulness vs Document Length", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / "faithfulness_vs_length.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'faithfulness_vs_length.png'}")
    plt.close()

    # Plot 4: Degradation Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    degradation_data = []
    for model_name, results in all_results.items():
        rouge1_rate = compute_degradation_rate(results, "rouge1_fmeasure")
        degradation_data.append({
            "Model": model_name,
            "Degradation Rate": -rouge1_rate * 100  # Convert to percentage, negative slope = degradation
        })

    df_deg = pd.DataFrame(degradation_data)
    df_deg = df_deg.sort_values("Degradation Rate")

    colors = ['#2ecc71' if x < 5 else '#f39c12' if x < 10 else '#e74c3c'
              for x in df_deg["Degradation Rate"]]

    ax.barh(df_deg["Model"], df_deg["Degradation Rate"], color=colors)
    ax.set_xlabel("Degradation Rate (%)", fontsize=12)
    ax.set_title("ROUGE-1 Degradation Rate by Model\n(Lower is Better)",
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels
    for i, v in enumerate(df_deg["Degradation Rate"]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va='center')

    plt.tight_layout()
    plt.savefig(output_dir / "degradation_rates.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'degradation_rates.png'}")
    plt.close()

    # Plot 5: Heatmap of all metrics
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for heatmap
    metric_keys = ["rouge1_fmeasure", "rouge2_fmeasure", "rougeL_fmeasure",
                   "bertscore_f1", "faithfulness_mean", "coverage_mean"]
    metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L",
                     "BERTScore", "Faithfulness", "Coverage"]

    heatmap_data = []
    row_labels = []

    for model_name in sorted(all_results.keys()):
        for bin_name in bins:
            scores = all_results[model_name].get(bin_name, {})
            row = [scores.get(k, 0) for k in metric_keys]
            heatmap_data.append(row)
            row_labels.append(f"{model_name}\n{bin_name}")

    heatmap_array = np.array(heatmap_data)

    im = ax.imshow(heatmap_array, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20)

    # Add values in cells
    for i in range(len(row_labels)):
        for j in range(len(metric_labels)):
            text = ax.text(j, i, f"{heatmap_array[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=7)

    ax.set_title("All Metrics by Model and Document Length",
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'metrics_heatmap.png'}")
    plt.close()


def generate_report(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
    dataset_name: str
):
    """Generate markdown report for length analysis.

    Args:
        all_results: Results for all models and bins
        output_dir: Output directory
        dataset_name: Name of dataset
    """
    report_path = output_dir / "length_analysis_report.md"

    # Get bins
    sample_results = next(iter(all_results.values()))
    bins = sorted(sample_results.keys())

    with open(report_path, "w") as f:
        f.write("# Document Length Analysis Report\n\n")
        f.write(f"**Dataset:** {dataset_name}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Overview\n\n")
        f.write("This report analyzes how model performance changes with document length, ")
        f.write("helping identify which approaches handle longer documents better.\n\n")

        f.write("## Length Bins\n\n")
        for bin_name in bins:
            sample_bin = next(iter(all_results.values()))
            num_samples = len(sample_bin.get(bin_name, {}))
            f.write(f"- **{bin_name}**: {num_samples} samples\n")
        f.write("\n")

        f.write("## ROUGE-1 Scores by Length\n\n")
        f.write("| Model | " + " | ".join(bins) + " | Degradation Rate |\n")
        f.write("|-------|" + "|".join(["-------"] * len(bins)) + "|------------------|\n")

        for model_name in sorted(all_results.keys()):
            results = all_results[model_name]
            scores = [results.get(b, {}).get("rouge1_fmeasure", 0) for b in bins]
            deg_rate = compute_degradation_rate(results, "rouge1_fmeasure")

            f.write(f"| {model_name} | ")
            f.write(" | ".join([f"{s:.4f}" for s in scores]))
            f.write(f" | {-deg_rate*100:.2f}% |\n")
        f.write("\n")

        f.write("## Key Findings\n\n")

        # Find best and worst performers
        rouge1_first_bin = {
            m: all_results[m].get(bins[0], {}).get("rouge1_fmeasure", 0)
            for m in all_results.keys()
        }
        rouge1_last_bin = {
            m: all_results[m].get(bins[-1], {}).get("rouge1_fmeasure", 0)
            for m in all_results.keys()
        }

        best_short = max(rouge1_first_bin.items(), key=lambda x: x[1])
        best_long = max(rouge1_last_bin.items(), key=lambda x: x[1])

        f.write(f"### Performance on Short Documents ({bins[0]})\n\n")
        f.write(f"- **Best:** {best_short[0]} (ROUGE-1: {best_short[1]:.4f})\n\n")

        f.write(f"### Performance on Long Documents ({bins[-1]})\n\n")
        f.write(f"- **Best:** {best_long[0]} (ROUGE-1: {best_long[1]:.4f})\n\n")

        # Degradation analysis
        degradation_rates = {
            m: compute_degradation_rate(all_results[m], "rouge1_fmeasure")
            for m in all_results.keys()
        }
        most_robust = min(degradation_rates.items(), key=lambda x: abs(x[1]))
        most_degraded = max(degradation_rates.items(), key=lambda x: abs(x[1]))

        f.write("### Robustness to Length\n\n")
        f.write(f"- **Most Robust:** {most_robust[0]} (degradation: {-most_robust[1]*100:.2f}%)\n")
        f.write(f"- **Most Affected:** {most_degraded[0]} (degradation: {-most_degraded[1]*100:.2f}%)\n\n")

        f.write("## Approach Comparison\n\n")

        approaches = {
            "Hierarchical": ["hierarchical"],
            "Sliding Window": ["sliding_window", "bart"],
            "Sparse Attention": ["longformer"],
        }

        for approach_name, model_list in approaches.items():
            f.write(f"### {approach_name}\n\n")
            for model in model_list:
                if model in all_results:
                    results = all_results[model]
                    deg_rate = compute_degradation_rate(results, "rouge1_fmeasure")
                    first_score = results.get(bins[0], {}).get("rouge1_fmeasure", 0)
                    last_score = results.get(bins[-1], {}).get("rouge1_fmeasure", 0)

                    f.write(f"**{model}:**\n")
                    f.write(f"- Short docs ({bins[0]}): {first_score:.4f}\n")
                    f.write(f"- Long docs ({bins[-1]}): {last_score:.4f}\n")
                    f.write(f"- Degradation: {-deg_rate*100:.2f}%\n")
                    f.write(f"- Performance drop: {(first_score - last_score)*100:.2f}%\n\n")

        f.write("## Visualizations\n\n")
        f.write("![ROUGE-1 vs Length](rouge1_vs_length.png)\n\n")
        f.write("![BERTScore vs Length](bertscore_vs_length.png)\n\n")
        f.write("![Faithfulness vs Length](faithfulness_vs_length.png)\n\n")
        f.write("![Degradation Rates](degradation_rates.png)\n\n")
        f.write("![Metrics Heatmap](metrics_heatmap.png)\n\n")

        f.write("## Recommendations\n\n")
        f.write("Based on the analysis:\n\n")

        if best_long[0] in ["longformer"]:
            f.write("- **For long documents (>10K tokens):** Use Longformer (sparse attention) ")
            f.write("for best performance and robustness.\n")

        if best_long[0] in ["hierarchical"]:
            f.write("- **For structured documents:** Hierarchical transformer provides good ")
            f.write("balance and handles document structure well.\n")

        f.write("- **For speed-critical applications:** Extractive methods (TextRank/LexRank) ")
        f.write("are fastest but sacrifice quality on longer documents.\n")

        if most_degraded[0] in ["bart", "sliding_window"]:
            f.write("- **Sliding window approaches:** Show significant degradation with length. ")
            f.write("Consider alternatives for very long documents.\n")

    print(f"\nReport saved to: {report_path}")


def main():
    """Main length analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze performance degradation with document length"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="arxiv",
        help="Dataset to use",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Total number of samples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/length_analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["textrank", "lexrank", "bart", "hierarchical", "longformer", "sliding_window"],
        help="Models to analyze",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["rouge", "bertscore", "faithfulness", "coverage"],
        help="Metrics to compute",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("DOCUMENT LENGTH ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load test data
    from scripts.compare_models import load_test_data

    samples = load_test_data(args.dataset, args.num_samples)

    if not samples:
        print("\nERROR: No test data found. Please run 'make preprocess' first.")
        sys.exit(1)

    # Define length bins (in tokens)
    bin_ranges = [
        (0, 6000),        # 0-6K
        (6000, 9000),     # 6-9K
        (9000, 12000),    # 9-12K
        (12000, 15000),   # 12-15K
        (15000, 999999),  # 15K+
    ]

    print("Binning samples by length...")
    binned_samples = bin_by_length(samples, bin_ranges)

    print("\nLength distribution:")
    for bin_name in sorted(binned_samples.keys()):
        print(f"  {bin_name}: {len(binned_samples[bin_name])} samples")

    # Analyze each model
    all_results = {}

    for model_name in args.models:
        results = analyze_model_by_length(
            model_name,
            binned_samples,
            args.metrics,
            args.device
        )

        if results:
            all_results[model_name] = results

    if not all_results:
        print("\nERROR: No models successfully analyzed.")
        sys.exit(1)

    # Save results
    results_file = output_dir / "length_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Create visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    plot_length_analysis(all_results, output_dir)

    # Generate report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")
    generate_report(all_results, output_dir, args.dataset)

    print(f"\n{'='*60}")
    print("LENGTH ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"View report: {output_dir / 'length_analysis_report.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
