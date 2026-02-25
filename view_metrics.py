"""
view_metrics.py — View and analyze RAG metrics

Utility script to view stored metrics, generate summaries, and export data.
"""

import argparse
import json
from engine.rag_metrics import MetricsStore


def print_summary(summary: dict):
    """Pretty print metrics summary."""
    print("\n" + "="*60)
    print("RAG METRICS SUMMARY")
    print("="*60)
    print(f"\nTotal Evaluations: {summary['count']}")
    
    if summary['count'] == 0:
        print("\nNo metrics data available.")
        return
    
    # BLEU Score
    if summary['bleu']['mean'] is not None:
        print("\n--- BLEU Score ---")
        print(f"  Mean:   {summary['bleu']['mean']:.3f}")
        print(f"  Median: {summary['bleu']['median']:.3f}")
        print(f"  Min:    {summary['bleu']['min']:.3f}")
        print(f"  Max:    {summary['bleu']['max']:.3f}")
    
    # Groundedness
    if summary['groundedness']['mean'] is not None:
        print("\n--- Groundedness Score ---")
        print(f"  Mean:   {summary['groundedness']['mean']:.3f}")
        print(f"  Median: {summary['groundedness']['median']:.3f}")
        print(f"  Min:    {summary['groundedness']['min']:.3f}")
        print(f"  Max:    {summary['groundedness']['max']:.3f}")
    
    # Hallucination Rate
    if summary['hallucination_rate']['mean'] is not None:
        print("\n--- Hallucination Rate ---")
        print(f"  Mean:   {summary['hallucination_rate']['mean']:.3f}")
        print(f"  Median: {summary['hallucination_rate']['median']:.3f}")
        print(f"  Min:    {summary['hallucination_rate']['min']:.3f}")
        print(f"  Max:    {summary['hallucination_rate']['max']:.3f}")
    
    print("\n" + "="*60 + "\n")


def print_recent_metrics(store: MetricsStore, n: int = 10):
    """Print the most recent metrics."""
    metrics = store.load_metrics()
    
    if not metrics:
        print("\nNo metrics data available.")
        return
    
    print(f"\n--- Last {min(n, len(metrics))} Evaluations ---\n")
    
    for i, m in enumerate(metrics[-n:], 1):
        print(f"{i}. Timestamp: {m.timestamp}")
        if m.bleu is not None:
            print(f"   BLEU: {m.bleu:.3f}")
        print(f"   Groundedness: {m.groundedness:.3f}")
        print(f"   Hallucination Rate: {m.hallucination_rate:.3f}")
        
        if m.metadata:
            print(f"   Metadata: {json.dumps(m.metadata, indent=6)}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze RAG evaluation metrics"
    )
    parser.add_argument(
        '--file',
        default='rag_metrics.jsonl',
        help='Path to metrics file (default: rag_metrics.jsonl)'
    )
    parser.add_argument(
        '--lesson',
        help='Filter by lesson name'
    )
    parser.add_argument(
        '--recent',
        type=int,
        metavar='N',
        help='Show N most recent evaluations'
    )
    parser.add_argument(
        '--export',
        metavar='OUTPUT',
        help='Export metrics to CSV file'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics (default action)'
    )
    
    args = parser.parse_args()
    
    # Initialize store
    store = MetricsStore(args.file)
    
    # Default to summary if no action specified
    if not any([args.recent, args.export]):
        args.summary = True
    
    # Show summary
    if args.summary:
        summary = store.get_summary(lesson_name=args.lesson)
        print_summary(summary)
    
    # Show recent metrics
    if args.recent:
        print_recent_metrics(store, args.recent)
    
    # Export to CSV
    if args.export:
        store.export_to_csv(args.export)
        print(f"\n✅ Metrics exported to {args.export}")


if __name__ == "__main__":
    main()
