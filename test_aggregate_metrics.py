"""
test_aggregate_metrics.py — Test aggregate metrics functionality

Quick test to verify aggregate metrics calculation works correctly.
"""

from engine.rag_metrics import evaluate_rag_response


def test_aggregate_metrics():
    """Test aggregate metrics for multiple questions."""
    
    # Simulate multiple quiz questions
    questions = [
        "What is the powerhouse of the cell?",
        "Where does cellular respiration occur?",
        "What molecule is produced by mitochondria?"
    ]
    
    # Combine all questions
    all_questions = " ".join(questions)
    
    # Context from knowledge graph
    context = """
    The mitochondria is the powerhouse of the cell. It produces ATP through 
    cellular respiration and oxidative phosphorylation. Cellular respiration 
    occurs in the mitochondria. ATP is the primary energy molecule.
    """
    
    # Calculate aggregate metrics
    metrics = evaluate_rag_response(
        generated=all_questions,
        retrieved_context=context,
        references=None,
        metadata={
            'lesson_name': 'Biology',
            'question_type': 'mcq',
            'num_questions': len(questions),
            'aggregate': True
        }
    )
    
    print("\n" + "="*60)
    print("AGGREGATE METRICS TEST")
    print("="*60)
    print(f"\nNumber of Questions: {len(questions)}")
    print(f"\nCombined Questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")
    
    print(f"\nAggregate Metrics:")
    print(f"  Groundedness:       {metrics.groundedness:.2%}")
    print(f"  Hallucination Rate: {metrics.hallucination_rate:.2%}")
    print(f"  BLEU Score:         {metrics.bleu if metrics.bleu else 'N/A'}")
    
    print(f"\nInterpretation:")
    if metrics.groundedness >= 0.7:
        print(f"  ✅ High groundedness - Questions well-supported by context")
    else:
        print(f"  ⚠️ Low groundedness - Questions may contain unsupported content")
    
    if metrics.hallucination_rate <= 0.3:
        print(f"  ✅ Low hallucination - Minimal fabricated content")
    else:
        print(f"  ⚠️ High hallucination - Significant fabricated content")
    
    print("\n" + "="*60 + "\n")
    
    # Assertions
    assert 0.0 <= metrics.groundedness <= 1.0, "Groundedness out of range"
    assert 0.0 <= metrics.hallucination_rate <= 1.0, "Hallucination rate out of range"
    assert metrics.groundedness > 0.5, "Expected reasonable groundedness"
    assert metrics.hallucination_rate < 0.5, "Expected low hallucination rate"
    
    print("✅ All tests passed!")


def test_aggregate_vs_individual():
    """Compare aggregate metrics vs individual question metrics."""
    
    questions = [
        "What is the mitochondria?",
        "What does quantum entanglement mean?"  # Hallucinated topic
    ]
    
    context = "The mitochondria is the powerhouse of the cell."
    
    # Individual metrics
    print("\n" + "="*60)
    print("AGGREGATE vs INDIVIDUAL COMPARISON")
    print("="*60)
    
    individual_metrics = []
    for i, q in enumerate(questions, 1):
        m = evaluate_rag_response(q, context)
        individual_metrics.append(m)
        print(f"\nQuestion {i}: {q}")
        print(f"  Groundedness: {m.groundedness:.2%}")
        print(f"  Hallucination: {m.hallucination_rate:.2%}")
    
    # Aggregate metrics
    all_questions = " ".join(questions)
    aggregate = evaluate_rag_response(all_questions, context)
    
    print(f"\nAggregate (Combined):")
    print(f"  Groundedness: {aggregate.groundedness:.2%}")
    print(f"  Hallucination: {aggregate.hallucination_rate:.2%}")
    
    # Calculate average of individual metrics
    avg_ground = sum(m.groundedness for m in individual_metrics) / len(individual_metrics)
    avg_hall = sum(m.hallucination_rate for m in individual_metrics) / len(individual_metrics)
    
    print(f"\nAverage of Individual Metrics:")
    print(f"  Avg Groundedness: {avg_ground:.2%}")
    print(f"  Avg Hallucination: {avg_hall:.2%}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_aggregate_metrics()
    test_aggregate_vs_individual()
