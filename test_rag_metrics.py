"""
test_rag_metrics.py — Unit tests for RAG evaluation metrics

Tests BLEU, Groundedness, and Hallucination Rate calculations.
"""

import pytest
from engine.rag_metrics import (
    tokenize,
    get_ngrams,
    split_sentences,
    calculate_bleu,
    calculate_groundedness,
    calculate_hallucination_rate,
    evaluate_rag_response,
    RAGMetrics,
    MetricsStore
)


# ─────────────────────────────────────────────
# Tokenization Tests
# ─────────────────────────────────────────────

def test_tokenize_basic():
    text = "Hello, World! This is a test."
    tokens = tokenize(text)
    assert tokens == ['hello', 'world', 'this', 'is', 'a', 'test']


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_get_ngrams():
    tokens = ['the', 'cat', 'sat', 'on', 'mat']
    
    unigrams = get_ngrams(tokens, 1)
    assert len(unigrams) == 5
    assert unigrams[0] == ('the',)
    
    bigrams = get_ngrams(tokens, 2)
    assert len(bigrams) == 4
    assert bigrams[0] == ('the', 'cat')
    
    trigrams = get_ngrams(tokens, 3)
    assert len(trigrams) == 3
    assert trigrams[0] == ('the', 'cat', 'sat')


def test_split_sentences():
    text = "First sentence. Second sentence! Third sentence?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence"


# ─────────────────────────────────────────────
# BLEU Score Tests
# ─────────────────────────────────────────────

def test_bleu_perfect_match():
    """Test BLEU with identical generated and reference text."""
    generated = "The cat sat on the mat"
    references = ["The cat sat on the mat"]
    
    bleu = calculate_bleu(generated, references)
    assert bleu == pytest.approx(1.0, abs=0.01)


def test_bleu_no_match():
    """Test BLEU with completely different texts."""
    generated = "The dog ran in the park"
    references = ["A bird flew over the ocean"]
    
    bleu = calculate_bleu(generated, references)
    assert bleu < 0.1


def test_bleu_partial_match():
    """Test BLEU with partial overlap."""
    generated = "The cat sat on the mat"
    references = ["The cat sat on a chair"]
    
    bleu = calculate_bleu(generated, references)
    assert 0.3 < bleu < 0.9


def test_bleu_multiple_references():
    """Test BLEU with multiple reference texts."""
    generated = "The cat sat on the mat"
    references = [
        "The cat sat on a chair",
        "The cat sat on the mat",
        "A cat was sitting on the mat"
    ]
    
    bleu = calculate_bleu(generated, references)
    assert bleu == pytest.approx(1.0, abs=0.01)


def test_bleu_empty_input():
    """Test BLEU with empty inputs."""
    assert calculate_bleu("", ["reference"]) == 0.0
    assert calculate_bleu("generated", []) == 0.0
    assert calculate_bleu("", []) == 0.0


def test_bleu_brevity_penalty():
    """Test BLEU brevity penalty for short generations."""
    generated = "The cat"
    references = ["The cat sat on the mat"]
    
    bleu = calculate_bleu(generated, references)
    # Should be penalized for being shorter
    assert bleu < 1.0


# ─────────────────────────────────────────────
# Groundedness Tests
# ─────────────────────────────────────────────

def test_groundedness_fully_grounded():
    """Test groundedness when all content is from context."""
    generated = "The mitochondria is the powerhouse of the cell."
    context = "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration."
    
    groundedness = calculate_groundedness(generated, context)
    assert groundedness == pytest.approx(1.0, abs=0.1)


def test_groundedness_not_grounded():
    """Test groundedness when content is not from context."""
    generated = "The Earth orbits around the Sun in an elliptical path."
    context = "The mitochondria is the powerhouse of the cell. It produces ATP."
    
    groundedness = calculate_groundedness(generated, context)
    assert groundedness < 0.3


def test_groundedness_partial():
    """Test groundedness with mixed grounded and ungrounded content."""
    generated = "The mitochondria produces ATP. The nucleus contains DNA. The cell wall is rigid."
    context = "The mitochondria produces ATP through respiration. The nucleus contains genetic material DNA."
    
    groundedness = calculate_groundedness(generated, context)
    assert 0.4 < groundedness < 0.9


def test_groundedness_empty_input():
    """Test groundedness with empty inputs."""
    assert calculate_groundedness("", "context") == 0.0
    assert calculate_groundedness("generated", "") == 0.0


def test_groundedness_threshold():
    """Test groundedness with different similarity thresholds."""
    generated = "The cat sat on the mat."
    context = "A cat was sitting on a mat in the room."
    
    # Lower threshold should give higher groundedness
    ground_low = calculate_groundedness(generated, context, similarity_threshold=0.3)
    ground_high = calculate_groundedness(generated, context, similarity_threshold=0.7)
    
    assert ground_low >= ground_high


# ─────────────────────────────────────────────
# Hallucination Rate Tests
# ─────────────────────────────────────────────

def test_hallucination_rate_no_hallucination():
    """Test hallucination rate when all words are in context."""
    generated = "The mitochondria produces ATP"
    context = "The mitochondria is the powerhouse of the cell and produces ATP through cellular respiration"
    
    rate = calculate_hallucination_rate(generated, context)
    assert rate == pytest.approx(0.0, abs=0.1)


def test_hallucination_rate_full_hallucination():
    """Test hallucination rate when no words are in context."""
    generated = "quantum entanglement superposition"
    context = "The mitochondria produces ATP through cellular respiration"
    
    rate = calculate_hallucination_rate(generated, context)
    assert rate == pytest.approx(1.0, abs=0.1)


def test_hallucination_rate_partial():
    """Test hallucination rate with mixed content."""
    generated = "The mitochondria produces quantum energy"
    context = "The mitochondria produces ATP through cellular respiration"
    
    rate = calculate_hallucination_rate(generated, context)
    assert 0.2 < rate < 0.8


def test_hallucination_rate_stopwords():
    """Test hallucination rate with and without stopwords."""
    generated = "The cat is on the mat"
    context = "A dog was in the park"
    
    # With stopwords ignored (default)
    rate_no_stop = calculate_hallucination_rate(generated, context, ignore_stopwords=True)
    
    # With stopwords included
    rate_with_stop = calculate_hallucination_rate(generated, context, ignore_stopwords=False)
    
    # Rate should be higher when stopwords are included
    assert rate_with_stop <= rate_no_stop


def test_hallucination_rate_empty_input():
    """Test hallucination rate with empty inputs."""
    assert calculate_hallucination_rate("", "context") == 0.0
    assert calculate_hallucination_rate("generated", "") >= 0.0


def test_hallucination_rate_identical_text():
    """Test hallucination rate when generated text is identical to context."""
    text = "The mitochondria is the powerhouse of the cell"
    
    rate = calculate_hallucination_rate(text, text)
    assert rate == pytest.approx(0.0, abs=0.01)


# ─────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────

def test_evaluate_rag_response_complete():
    """Test complete RAG evaluation with all metrics."""
    generated = "The mitochondria produces ATP through cellular respiration"
    context = "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration and oxidative phosphorylation."
    references = ["The mitochondria produces ATP through cellular respiration"]
    
    metrics = evaluate_rag_response(
        generated=generated,
        retrieved_context=context,
        references=references,
        metadata={'lesson': 'biology', 'type': 'quiz'}
    )
    
    assert metrics.bleu is not None
    assert metrics.groundedness is not None
    assert metrics.hallucination_rate is not None
    assert metrics.bleu > 0.8
    assert metrics.groundedness > 0.7
    assert metrics.hallucination_rate < 0.3
    assert metrics.metadata['lesson'] == 'biology'


def test_evaluate_rag_response_no_references():
    """Test RAG evaluation without reference texts."""
    generated = "The mitochondria produces ATP"
    context = "The mitochondria is the powerhouse of the cell"
    
    metrics = evaluate_rag_response(
        generated=generated,
        retrieved_context=context
    )
    
    assert metrics.bleu is None  # No references provided
    assert metrics.groundedness is not None
    assert metrics.hallucination_rate is not None


# ─────────────────────────────────────────────
# RAGMetrics Class Tests
# ─────────────────────────────────────────────

def test_rag_metrics_to_dict():
    """Test RAGMetrics serialization to dictionary."""
    metrics = RAGMetrics(
        bleu=0.85,
        groundedness=0.92,
        hallucination_rate=0.15,
        metadata={'lesson': 'test'}
    )
    
    data = metrics.to_dict()
    assert data['bleu'] == 0.85
    assert data['groundedness'] == 0.92
    assert data['hallucination_rate'] == 0.15
    assert data['metadata']['lesson'] == 'test'
    assert 'timestamp' in data


def test_rag_metrics_from_dict():
    """Test RAGMetrics deserialization from dictionary."""
    data = {
        'bleu': 0.85,
        'groundedness': 0.92,
        'hallucination_rate': 0.15,
        'metadata': {'lesson': 'test'}
    }
    
    metrics = RAGMetrics.from_dict(data)
    assert metrics.bleu == 0.85
    assert metrics.groundedness == 0.92
    assert metrics.hallucination_rate == 0.15


# ─────────────────────────────────────────────
# MetricsStore Tests
# ─────────────────────────────────────────────

def test_metrics_store_add_and_load(tmp_path):
    """Test adding and loading metrics from store."""
    filepath = tmp_path / "test_metrics.jsonl"
    store = MetricsStore(str(filepath))
    
    metrics1 = RAGMetrics(bleu=0.8, groundedness=0.9, hallucination_rate=0.1)
    metrics2 = RAGMetrics(bleu=0.7, groundedness=0.85, hallucination_rate=0.2)
    
    store.add_metrics(metrics1)
    store.add_metrics(metrics2)
    
    loaded = store.load_metrics()
    assert len(loaded) == 2
    assert loaded[0].bleu == 0.8
    assert loaded[1].bleu == 0.7


def test_metrics_store_summary(tmp_path):
    """Test metrics summary statistics."""
    filepath = tmp_path / "test_metrics.jsonl"
    store = MetricsStore(str(filepath))
    
    # Add multiple metrics
    for i in range(5):
        metrics = RAGMetrics(
            bleu=0.7 + i * 0.05,
            groundedness=0.8 + i * 0.03,
            hallucination_rate=0.2 - i * 0.02,
            metadata={'lesson': 'test'}
        )
        store.add_metrics(metrics)
    
    summary = store.get_summary()
    assert summary['count'] == 5
    assert summary['bleu']['mean'] == pytest.approx(0.8, abs=0.1)
    assert summary['groundedness']['mean'] == pytest.approx(0.86, abs=0.1)
    assert summary['hallucination_rate']['mean'] == pytest.approx(0.16, abs=0.1)


def test_metrics_store_summary_filtered(tmp_path):
    """Test metrics summary filtered by lesson."""
    filepath = tmp_path / "test_metrics.jsonl"
    store = MetricsStore(str(filepath))
    
    # Add metrics for different lessons
    store.add_metrics(RAGMetrics(bleu=0.8, groundedness=0.9, hallucination_rate=0.1,
                                  metadata={'lesson_name': 'biology'}))
    store.add_metrics(RAGMetrics(bleu=0.7, groundedness=0.85, hallucination_rate=0.2,
                                  metadata={'lesson_name': 'physics'}))
    store.add_metrics(RAGMetrics(bleu=0.75, groundedness=0.88, hallucination_rate=0.15,
                                  metadata={'lesson_name': 'biology'}))
    
    summary = store.get_summary(lesson_name='biology')
    assert summary['count'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
