"""
demo_metrics.py — Demonstration of RAG metrics

Shows how to use the metrics module with example data.
"""

from engine.rag_metrics import (
    calculate_bleu,
    calculate_groundedness,
    calculate_hallucination_rate,
    evaluate_rag_response,
    MetricsStore
)


def demo_bleu():
    """Demonstrate BLEU score calculation."""
    print("\n" + "="*60)
    print("BLEU SCORE DEMONSTRATION")
    print("="*60)
    
    generated = "The mitochondria is the powerhouse of the cell"
    references = [
        "The mitochondria is the powerhouse of the cell",
        "Mitochondria are the powerhouse of cells"
    ]
    
    bleu = calculate_bleu(generated, references)
    print(f"\nGenerated: {generated}")
    print(f"References: {references}")
    print(f"BLEU Score: {bleu:.3f}")
    
    # Partial match example
    generated2 = "The mitochondria produces ATP"
    bleu2 = calculate_bleu(generated2, references)
    print(f"\nGenerated: {generated2}")
    print(f"BLEU Score: {bleu2:.3f} (lower due to less overlap)")


def demo_groundedness():
    """Demonstrate groundedness measurement."""
    print("\n" + "="*60)
    print("GROUNDEDNESS DEMONSTRATION")
    print("="*60)
    
    context = """
    The mitochondria is the powerhouse of the cell. It produces ATP through 
    cellular respiration. The process involves the electron transport chain 
    and oxidative phosphorylation.
    """
    
    # Fully grounded response
    generated1 = "The mitochondria produces ATP through cellular respiration."
    ground1 = calculate_groundedness(generated1, context)
    print(f"\nGenerated: {generated1}")
    print(f"Groundedness: {ground1:.3f} (fully grounded)")
    
    # Partially grounded response
    generated2 = "The mitochondria produces ATP. It also stores genetic information."
    ground2 = calculate_groundedness(generated2, context)
    print(f"\nGenerated: {generated2}")
    print(f"Groundedness: {ground2:.3f} (partially grounded - second sentence is hallucinated)")
    
    # Not grounded response
    generated3 = "The nucleus contains DNA and controls cell activities."
    ground3 = calculate_groundedness(generated3, context)
    print(f"\nGenerated: {generated3}")
    print(f"Groundedness: {ground3:.3f} (not grounded in context)")


def demo_hallucination_rate():
    """Demonstrate hallucination rate calculation."""
    print("\n" + "="*60)
    print("HALLUCINATION RATE DEMONSTRATION")
    print("="*60)
    
    context = """
    The mitochondria is the powerhouse of the cell. It produces ATP through 
    cellular respiration and oxidative phosphorylation.
    """
    
    # No hallucination
    generated1 = "The mitochondria produces ATP through cellular respiration"
    rate1 = calculate_hallucination_rate(generated1, context)
    print(f"\nGenerated: {generated1}")
    print(f"Hallucination Rate: {rate1:.3f} (no hallucinations)")
    
    # Some hallucination
    generated2 = "The mitochondria produces quantum energy through fusion"
    rate2 = calculate_hallucination_rate(generated2, context)
    print(f"\nGenerated: {generated2}")
    print(f"Hallucination Rate: {rate2:.3f} (contains hallucinated terms)")
    
    # High hallucination
    generated3 = "Photosynthesis converts sunlight into glucose in chloroplasts"
    rate3 = calculate_hallucination_rate(generated3, context)
    print(f"\nGenerated: {generated3}")
    print(f"Hallucination Rate: {rate3:.3f} (completely different topic)")


def demo_full_evaluation():
    """Demonstrate full RAG evaluation."""
    print("\n" + "="*60)
    print("FULL RAG EVALUATION DEMONSTRATION")
    print("="*60)
    
    context = """
    Photosynthesis is the process by which plants convert light energy into 
    chemical energy. It occurs in chloroplasts and produces glucose and oxygen. 
    The process requires carbon dioxide, water, and sunlight.
    """
    
    generated = "Photosynthesis converts light energy into chemical energy in chloroplasts."
    references = ["Photosynthesis is the process of converting light energy to chemical energy."]
    
    metrics = evaluate_rag_response(
        generated=generated,
        retrieved_context=context,
        references=references,
        metadata={'topic': 'biology', 'type': 'demo'}
    )
    
    print(f"\nContext: {context.strip()}")
    print(f"\nGenerated: {generated}")
    print(f"\nMetrics:")
    print(f"  BLEU Score:         {metrics.bleu:.3f}")
    print(f"  Groundedness:       {metrics.groundedness:.3f}")
    print(f"  Hallucination Rate: {metrics.hallucination_rate:.3f}")
    print(f"\nInterpretation:")
    print(f"  - High BLEU ({metrics.bleu:.3f}) indicates good similarity to reference")
    print(f"  - High Groundedness ({metrics.groundedness:.3f}) means content is well-supported")
    print(f"  - Low Hallucination ({metrics.hallucination_rate:.3f}) shows minimal fabricated content")


def demo_metrics_storage():
    """Demonstrate metrics storage and retrieval."""
    print("\n" + "="*60)
    print("METRICS STORAGE DEMONSTRATION")
    print("="*60)
    
    store = MetricsStore("demo_metrics.jsonl")
    
    # Generate some sample metrics
    contexts = [
        "The Earth orbits the Sun in an elliptical path.",
        "Water boils at 100 degrees Celsius at sea level.",
        "DNA contains genetic information in living organisms."
    ]
    
    responses = [
        "The Earth orbits the Sun.",
        "Water boils at 100 degrees.",
        "DNA stores genetic information."
    ]
    
    print("\nStoring metrics for 3 evaluations...")
    for i, (context, response) in enumerate(zip(contexts, responses), 1):
        metrics = evaluate_rag_response(
            generated=response,
            retrieved_context=context,
            metadata={'demo_id': i, 'lesson': 'science'}
        )
        store.add_metrics(metrics)
        print(f"  {i}. Stored metrics for: '{response[:40]}...'")
    
    # Get summary
    print("\nRetrieving summary statistics...")
    summary = store.get_summary()
    
    print(f"\nSummary:")
    print(f"  Total evaluations: {summary['count']}")
    print(f"  Avg Groundedness: {summary['groundedness']['mean']:.3f}")
    print(f"  Avg Hallucination Rate: {summary['hallucination_rate']['mean']:.3f}")
    
    print(f"\n✅ Metrics saved to demo_metrics.jsonl")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("RAG METRICS DEMONSTRATION SUITE")
    print("="*60)
    
    demo_bleu()
    demo_groundedness()
    demo_hallucination_rate()
    demo_full_evaluation()
    demo_metrics_storage()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nTo view stored metrics, run:")
    print("  python view_metrics.py --file demo_metrics.jsonl")
    print("\nTo run tests:")
    print("  pytest test_rag_metrics.py -v")
    print()


if __name__ == "__main__":
    main()
