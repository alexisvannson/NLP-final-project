"""Analyze how well models handle document structure and sections.

This script evaluates whether models capture information from all important
document sections and how section-aware approaches perform.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.hierarchical_transformer import HierarchicalTransformerSummarizer
from src.evaluation import SectionCoverageEvaluator


def load_test_data(dataset_name: str, num_samples: int = 50) -> List[Dict]:
    """Load test data."""
    data_path = Path("data/processed") / dataset_name / "test.json"

    if not data_path.exists():
        print(f"Warning: Test data not found at {data_path}")
        return []

    with open(data_path, "r") as f:
        data = json.load(f)

    return data[:num_samples]


def analyze_section_structure(samples: List[Dict]) -> Dict:
    """Analyze section structure in documents.

    Args:
        samples: Test samples

    Returns:
        Dictionary with section statistics
    """
    print("\nAnalyzing document section structure...")

    from models.hierarchical_transformer import HierarchicalTransformerSummarizer

    # Create summarizer just to access section detection
    summarizer = HierarchicalTransformerSummarizer()

    all_sections = []
    section_counts = []
    documents_with_sections = 0

    for sample in tqdm(samples, desc="Detecting sections"):
        source = sample.get("article", sample.get("document", ""))

        sections = summarizer.detect_sections(source)
        section_titles = [s['title'] for s in sections]

        all_sections.extend([s.lower() for s in section_titles])
        section_counts.append(len(sections))

        if len(sections) > 1:
            documents_with_sections += 1

    # Analyze section distribution
    section_counter = Counter(all_sections)
    common_sections = section_counter.most_common(20)

    return {
        'total_documents': len(samples),
        'documents_with_sections': documents_with_sections,
        'percent_with_sections': (documents_with_sections / len(samples)) * 100,
        'avg_sections_per_doc': np.mean(section_counts),
        'median_sections': np.median(section_counts),
        'max_sections': max(section_counts),
        'common_sections': common_sections,
        'all_section_counts': section_counts
    }


def compare_section_aware_models(samples: List[Dict], device: str = "cpu") -> Dict:
    """Compare regular vs section-aware summarization.

    Args:
        samples: Test samples
        device: Device to use

    Returns:
        Comparison results
    """
    print("\nComparing section-aware vs regular summarization...")

    try:
        summarizer = HierarchicalTransformerSummarizer(device=device)
    except Exception as e:
        print(f"Could not load hierarchical model: {e}")
        return None

    evaluator = SectionCoverageEvaluator()

    regular_summaries = []
    section_aware_summaries = []
    sources = []

    for sample in tqdm(samples, desc="Generating summaries"):
        source = sample.get("article", sample.get("document", ""))
        sources.append(source)

        try:
            # Regular summarization
            regular_summary = summarizer.summarize(source)
            regular_summaries.append(regular_summary)

            # Section-aware summarization
            section_summary = summarizer.summarize_with_sections(source)
            section_aware_summaries.append(section_summary)
        except Exception as e:
            print(f"Error generating summary: {e}")
            regular_summaries.append("")
            section_aware_summaries.append("")

    # Evaluate section coverage
    print("\nEvaluating regular summaries...")
    regular_scores = evaluator.evaluate(regular_summaries, sources)

    print("Evaluating section-aware summaries...")
    section_aware_scores = evaluator.evaluate(section_aware_summaries, sources)

    return {
        'regular': regular_scores,
        'section_aware': section_aware_scores,
        'improvement': {
            'coverage': section_aware_scores['section_coverage_mean'] -
                       regular_scores['section_coverage_mean'],
            'sections_covered': section_aware_scores['avg_sections_covered'] -
                               regular_scores['avg_sections_covered']
        }
    }


def analyze_section_coverage_by_model(
    samples: List[Dict],
    models: List[str],
    device: str = "cpu"
) -> Dict[str, Dict]:
    """Analyze section coverage for multiple models.

    Args:
        samples: Test samples
        models: List of model names
        device: Device to use

    Returns:
        Dictionary mapping model names to coverage scores
    """
    from models.baseline_abstractive import BARTChunkSummarizer
    from models.baseline_extractive import LexRankSummarizer, TextRankSummarizer
    from models.hierarchical_transformer import HierarchicalTransformerSummarizer
    from models.longformer_summarizer import LongformerSummarizer
    from models.sliding_window import SlidingWindowSummarizer

    print("\nAnalyzing section coverage by model...")

    results = {}
    evaluator = SectionCoverageEvaluator()

    for model_name in models:
        print(f"\n{model_name.upper()}")

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
                print(f"Unknown model: {model_name}")
                continue
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            continue

        # Generate summaries
        predictions = []
        sources = []

        for sample in tqdm(samples, desc=f"Summarizing"):
            source = sample.get("article", sample.get("document", ""))
            sources.append(source)

            try:
                prediction = model.summarize(source)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error: {e}")
                predictions.append("")

        # Evaluate
        scores = evaluator.evaluate(predictions, sources)
        results[model_name] = scores

        print(f"  Section Coverage: {scores['section_coverage_mean']:.4f}")
        print(f"  Avg Sections Covered: {scores['avg_sections_covered']:.2f}")

    return results


def plot_section_analysis(
    structure_stats: Dict,
    model_comparison: Dict,
    output_dir: Path
):
    """Create visualizations for section analysis.

    Args:
        structure_stats: Section structure statistics
        model_comparison: Model comparison results
        output_dir: Output directory
    """
    sns.set_style("whitegrid")

    # Plot 1: Section distribution in documents
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Number of sections histogram
    ax = axes[0]
    ax.hist(structure_stats['all_section_counts'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Number of Sections")
    ax.set_ylabel("Number of Documents")
    ax.set_title("Distribution of Sections per Document")
    ax.axvline(structure_stats['avg_sections_per_doc'], color='red',
               linestyle='--', label=f"Mean: {structure_stats['avg_sections_per_doc']:.1f}")
    ax.legend()

    # Common sections bar chart
    ax = axes[1]
    common_sections = structure_stats['common_sections'][:10]
    sections, counts = zip(*common_sections) if common_sections else ([], [])
    ax.barh(range(len(sections)), counts)
    ax.set_yticks(range(len(sections)))
    ax.set_yticklabels(sections)
    ax.set_xlabel("Frequency")
    ax.set_title("Most Common Section Titles")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "section_structure.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'section_structure.png'}")
    plt.close()

    # Plot 2: Model comparison on section coverage
    if model_comparison:
        fig, ax = plt.subplots(figsize=(12, 6))

        models = sorted(model_comparison.keys())
        coverage_scores = [model_comparison[m]['section_coverage_mean'] for m in models]
        sections_covered = [model_comparison[m]['avg_sections_covered'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, coverage_scores, width, label='Coverage Ratio', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, sections_covered, width, label='Sections Covered',
                alpha=0.8, color='orange')

        ax.set_xlabel('Model')
        ax.set_ylabel('Section Coverage Ratio', color='blue')
        ax2.set_ylabel('Avg Sections Covered', color='orange')
        ax.set_title('Section Coverage by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(output_dir / "section_coverage_comparison.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'section_coverage_comparison.png'}")
        plt.close()


def generate_report(
    structure_stats: Dict,
    section_aware_comparison: Dict,
    model_comparison: Dict,
    output_dir: Path,
    dataset_name: str
):
    """Generate markdown report.

    Args:
        structure_stats: Section structure statistics
        section_aware_comparison: Section-aware vs regular comparison
        model_comparison: Model comparison results
        output_dir: Output directory
        dataset_name: Dataset name
    """
    report_path = output_dir / "section_analysis_report.md"

    with open(report_path, "w") as f:
        f.write("# Document Section Structure Analysis\n\n")
        f.write(f"**Dataset:** {dataset_name}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Document Structure Statistics\n\n")
        f.write(f"- **Total documents analyzed:** {structure_stats['total_documents']}\n")
        f.write(f"- **Documents with sections:** {structure_stats['documents_with_sections']} ")
        f.write(f"({structure_stats['percent_with_sections']:.1f}%)\n")
        f.write(f"- **Average sections per document:** {structure_stats['avg_sections_per_doc']:.2f}\n")
        f.write(f"- **Median sections:** {structure_stats['median_sections']:.0f}\n")
        f.write(f"- **Maximum sections:** {structure_stats['max_sections']}\n\n")

        f.write("### Most Common Section Titles\n\n")
        f.write("| Section Title | Frequency |\n")
        f.write("|---------------|----------|\n")
        for section, count in structure_stats['common_sections'][:10]:
            f.write(f"| {section} | {count} |\n")
        f.write("\n")

        if section_aware_comparison:
            f.write("## Section-Aware vs Regular Summarization\n\n")
            regular = section_aware_comparison['regular']
            section_aware = section_aware_comparison['section_aware']
            improvement = section_aware_comparison['improvement']

            f.write("| Metric | Regular | Section-Aware | Improvement |\n")
            f.write("|--------|---------|---------------|-------------|\n")
            f.write(f"| Section Coverage | {regular['section_coverage_mean']:.4f} | ")
            f.write(f"{section_aware['section_coverage_mean']:.4f} | ")
            f.write(f"+{improvement['coverage']:.4f} |\n")
            f.write(f"| Avg Sections Covered | {regular['avg_sections_covered']:.2f} | ")
            f.write(f"{section_aware['avg_sections_covered']:.2f} | ")
            f.write(f"+{improvement['sections_covered']:.2f} |\n\n")

            if improvement['coverage'] > 0:
                f.write("✅ **Section-aware summarization improves coverage** by ")
                f.write(f"{improvement['coverage']*100:.1f}%\n\n")
            else:
                f.write("⚠️ Section-aware approach shows minimal improvement\n\n")

        if model_comparison:
            f.write("## Section Coverage by Model\n\n")
            f.write("| Model | Coverage Ratio | Avg Sections Covered |\n")
            f.write("|-------|----------------|---------------------|\n")
            for model in sorted(model_comparison.keys()):
                scores = model_comparison[model]
                f.write(f"| {model} | {scores['section_coverage_mean']:.4f} | ")
                f.write(f"{scores['avg_sections_covered']:.2f} |\n")
            f.write("\n")

            # Find best model
            best_model = max(model_comparison.items(),
                           key=lambda x: x[1]['section_coverage_mean'])
            f.write(f"**Best section coverage:** {best_model[0]} ")
            f.write(f"({best_model[1]['section_coverage_mean']:.4f})\n\n")

        f.write("## Key Findings\n\n")
        if structure_stats['percent_with_sections'] > 50:
            f.write(f"- {structure_stats['percent_with_sections']:.0f}% of documents have explicit sections\n")
            f.write("- Section-aware approaches are recommended for this dataset\n")
        else:
            f.write(f"- Only {structure_stats['percent_with_sections']:.0f}% of documents have explicit sections\n")
            f.write("- Section detection may be less effective on this dataset\n")

        if model_comparison:
            hierarchical_score = model_comparison.get('hierarchical', {}).get('section_coverage_mean', 0)
            avg_score = np.mean([m['section_coverage_mean'] for m in model_comparison.values()])

            if hierarchical_score > avg_score * 1.1:
                f.write("- Hierarchical model shows significantly better section coverage\n")

        f.write("\n## Visualizations\n\n")
        f.write("![Section Structure](section_structure.png)\n\n")
        if model_comparison:
            f.write("![Section Coverage Comparison](section_coverage_comparison.png)\n\n")

    print(f"\nReport saved to: {report_path}")


def main():
    """Main section analysis script."""
    parser = argparse.ArgumentParser(description="Analyze document section structure")
    parser.add_argument("--dataset", type=str, default="arxiv", help="Dataset to use")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="results/section_analysis",
                       help="Output directory")
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare section coverage across models")
    parser.add_argument("--models", nargs="+",
                       default=["textrank", "hierarchical", "longformer"],
                       help="Models to compare")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SECTION STRUCTURE ANALYSIS")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load data
    samples = load_test_data(args.dataset, args.num_samples)
    if not samples:
        print("ERROR: No test data found. Please run 'make preprocess' first.")
        sys.exit(1)

    # Analyze section structure
    structure_stats = analyze_section_structure(samples)

    print(f"\nStructure Statistics:")
    print(f"  Documents with sections: {structure_stats['documents_with_sections']}/{structure_stats['total_documents']} ({structure_stats['percent_with_sections']:.1f}%)")
    print(f"  Avg sections per doc: {structure_stats['avg_sections_per_doc']:.2f}")

    # Compare section-aware vs regular (if hierarchical model works)
    section_aware_comparison = compare_section_aware_models(samples[:min(20, len(samples))], args.device)

    # Compare models if requested
    model_comparison = None
    if args.compare_models:
        model_comparison = analyze_section_coverage_by_model(
            samples,
            args.models,
            args.device
        )

    # Create visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    plot_section_analysis(structure_stats, model_comparison, output_dir)

    # Generate report
    print(f"\n{'='*60}")
    print("GENERATING REPORT")
    print(f"{'='*60}")
    generate_report(
        structure_stats,
        section_aware_comparison,
        model_comparison,
        output_dir,
        args.dataset
    )

    # Save results
    results_file = output_dir / "section_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump({
            'structure_stats': {k: v for k, v in structure_stats.items()
                              if k != 'all_section_counts'},
            'section_aware_comparison': section_aware_comparison,
            'model_comparison': model_comparison
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SECTION ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"View report: {output_dir / 'section_analysis_report.md'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
