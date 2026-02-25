"""
rag_metrics.py — RAG Evaluation Metrics

Implements BLEU, Groundedness, and Hallucination Rate metrics
to measure quality and reliability of RAG system outputs.
"""

from __future__ import annotations
import re
import math
import json
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime


# ─────────────────────────────────────────────
# Tokenization & Preprocessing
# ─────────────────────────────────────────────

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
}


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Tokenize text into words, optionally converting to lowercase."""
    if not text:
        return []
    
    # Remove punctuation and split on whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    
    if lowercase:
        tokens = [t.lower() for t in tokens]
    
    return [t for t in tokens if t]


def get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    """Extract n-grams from a list of tokens."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────
# BLEU Score Implementation
# ─────────────────────────────────────────────

def calculate_bleu(
    generated: str,
    references: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate BLEU score for generated text against reference texts.
    
    Args:
        generated: The generated response text
        references: List of reference texts (ground truth)
        max_n: Maximum n-gram size (default: 4 for BLEU-4)
        weights: Weights for each n-gram precision (default: uniform)
    
    Returns:
        BLEU score between 0.0 and 1.0
    """
    if not generated or not references:
        return 0.0
    
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Tokenize
    gen_tokens = tokenize(generated)
    ref_tokens_list = [tokenize(ref) for ref in references]
    
    if not gen_tokens:
        return 0.0
    
    # Calculate modified precision for each n-gram size
    precisions = []
    for n in range(1, max_n + 1):
        gen_ngrams = get_ngrams(gen_tokens, n)
        if not gen_ngrams:
            precisions.append(0.0)
            continue
        
        # Count generated n-grams
        gen_counts = Counter(gen_ngrams)
        
        # Get maximum reference counts for each n-gram
        max_ref_counts: Dict[tuple, int] = defaultdict(int)
        for ref_tokens in ref_tokens_list:
            ref_ngrams = get_ngrams(ref_tokens, n)
            ref_counts = Counter(ref_ngrams)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        
        # Calculate clipped counts
        clipped_counts = sum(
            min(gen_counts[ngram], max_ref_counts[ngram])
            for ngram in gen_counts
        )
        
        # Modified precision
        precision = clipped_counts / len(gen_ngrams) if gen_ngrams else 0.0
        precisions.append(precision)
    
    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
        geo_mean = math.exp(log_sum)
    
    # Brevity penalty
    gen_len = len(gen_tokens)
    ref_lens = [len(ref_tokens) for ref_tokens in ref_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - gen_len))
    
    if gen_len >= closest_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - closest_ref_len / gen_len) if gen_len > 0 else 0.0
    
    bleu_score = bp * geo_mean
    return min(max(bleu_score, 0.0), 1.0)


# ─────────────────────────────────────────────
# Groundedness Measurement
# ─────────────────────────────────────────────

def calculate_groundedness(
    generated: str,
    retrieved_context: str,
    similarity_threshold: float = 0.3  # Lowered from 0.5 for better detection
) -> float:
    """
    Measure how well generated text is grounded in retrieved sources.
    
    Args:
        generated: The generated response text
        retrieved_context: The retrieved context/source documents
        similarity_threshold: Token overlap threshold for sentence support
    
    Returns:
        Groundedness score between 0.0 and 1.0
    """
    if not generated:
        return 0.0
    
    if not retrieved_context:
        return 0.0
    
    # Split generated text into sentences
    gen_sentences = split_sentences(generated)
    if not gen_sentences:
        return 0.0
    
    # Tokenize context
    context_tokens = set(tokenize(retrieved_context))
    if not context_tokens:
        return 0.0
    
    # Check each sentence for grounding
    supported_count = 0
    for sentence in gen_sentences:
        sent_tokens = set(tokenize(sentence))
        if not sent_tokens:
            continue
        
        # Calculate token overlap ratio (what % of sentence tokens are in context)
        intersection = sent_tokens & context_tokens
        
        # Use ratio of sentence tokens found in context
        if sent_tokens:
            overlap_ratio = len(intersection) / len(sent_tokens)
            if overlap_ratio >= similarity_threshold:
                supported_count += 1
    
    groundedness = supported_count / len(gen_sentences)
    return min(max(groundedness, 0.0), 1.0)


# ─────────────────────────────────────────────
# Hallucination Rate Calculation
# ─────────────────────────────────────────────

def calculate_hallucination_rate(
    generated: str,
    retrieved_context: str,
    ignore_stopwords: bool = True,
    fuzzy_match: bool = True  # NEW: Enable fuzzy matching for technical terms
) -> float:
    """
    Calculate hallucination rate: ratio of words not found in retrieved docs.
    
    Formula: |Words in Response - Words in Retrieved Docs| / |Words in Response|
    
    Args:
        generated: The generated response text
        retrieved_context: The retrieved context/source documents
        ignore_stopwords: Whether to ignore common stop words
        fuzzy_match: Whether to use fuzzy matching for technical terms
    
    Returns:
        Hallucination rate between 0.0 and 1.0 (0.0 = no hallucinations)
    """
    if not generated:
        return 0.0
    
    # Tokenize both texts
    gen_tokens = tokenize(generated)
    context_tokens = set(tokenize(retrieved_context))
    
    if not gen_tokens:
        return 0.0
    
    # Filter stop words if requested
    if ignore_stopwords:
        gen_tokens = [t for t in gen_tokens if t not in STOP_WORDS]
        context_tokens = {t for t in context_tokens if t not in STOP_WORDS}
    
    if not gen_tokens:
        return 0.0
    
    # Count unique tokens in generated text not in context
    gen_unique = set(gen_tokens)
    
    if fuzzy_match:
        # For each generated token, check if it or a similar form exists in context
        hallucinated = set()
        for gen_token in gen_unique:
            # Exact match
            if gen_token in context_tokens:
                continue
            
            # Check for partial matches (substring matching for compound words)
            found = False
            for ctx_token in context_tokens:
                # Check if gen_token is part of a context token or vice versa
                if len(gen_token) >= 4 and len(ctx_token) >= 4:
                    if gen_token in ctx_token or ctx_token in gen_token:
                        found = True
                        break
            
            if not found:
                hallucinated.add(gen_token)
    else:
        hallucinated = gen_unique - context_tokens
    
    hallucination_rate = len(hallucinated) / len(gen_unique)
    return min(max(hallucination_rate, 0.0), 1.0)


# ─────────────────────────────────────────────
# Metrics Aggregation & Storage
# ─────────────────────────────────────────────

class RAGMetrics:
    """Container for RAG evaluation metrics."""
    
    def __init__(
        self,
        bleu: Optional[float] = None,
        groundedness: Optional[float] = None,
        hallucination_rate: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.bleu = bleu
        self.groundedness = groundedness
        self.hallucination_rate = hallucination_rate
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'bleu': self.bleu,
            'groundedness': self.groundedness,
            'hallucination_rate': self.hallucination_rate,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGMetrics':
        """Create RAGMetrics from dictionary."""
        return cls(
            bleu=data.get('bleu'),
            groundedness=data.get('groundedness'),
            hallucination_rate=data.get('hallucination_rate'),
            metadata=data.get('metadata', {})
        )


def evaluate_rag_response(
    generated: str,
    retrieved_context: str,
    references: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> RAGMetrics:
    """
    Evaluate a RAG response with all metrics.
    
    Args:
        generated: The generated response text
        retrieved_context: The retrieved context/source documents
        references: Optional reference texts for BLEU calculation
        metadata: Optional metadata (lesson name, question type, etc.)
    
    Returns:
        RAGMetrics object with all calculated metrics
    """
    # Calculate BLEU if references provided
    bleu = None
    if references:
        bleu = calculate_bleu(generated, references)
    
    # Calculate Groundedness
    groundedness = calculate_groundedness(generated, retrieved_context)
    
    # Calculate Hallucination Rate
    hallucination_rate = calculate_hallucination_rate(generated, retrieved_context)
    
    return RAGMetrics(
        bleu=bleu,
        groundedness=groundedness,
        hallucination_rate=hallucination_rate,
        metadata=metadata
    )


# ─────────────────────────────────────────────
# Metrics Storage & Reporting
# ─────────────────────────────────────────────

class MetricsStore:
    """Store and aggregate RAG metrics."""
    
    def __init__(self, filepath: str = "rag_metrics.jsonl"):
        self.filepath = filepath
        self.metrics_cache: List[RAGMetrics] = []
    
    def add_metrics(self, metrics: RAGMetrics):
        """Add metrics to store and persist to file."""
        self.metrics_cache.append(metrics)
        
        # Append to JSONL file
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f)
                f.write('\n')
        except Exception as e:
            print(f"Warning: Failed to persist metrics: {e}")
    
    def load_metrics(self) -> List[RAGMetrics]:
        """Load all metrics from file."""
        metrics = []
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        metrics.append(RAGMetrics.from_dict(data))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Failed to load metrics: {e}")
        
        return metrics
    
    def get_summary(self, lesson_name: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregate statistics for metrics."""
        metrics = self.load_metrics()
        
        # Filter by lesson if specified
        if lesson_name:
            metrics = [
                m for m in metrics
                if m.metadata.get('lesson_name') == lesson_name
            ]
        
        if not metrics:
            return {
                'count': 0,
                'bleu': None,
                'groundedness': None,
                'hallucination_rate': None
            }
        
        # Calculate statistics
        def calc_stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {'mean': None, 'median': None, 'min': None, 'max': None}
            
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            
            return {
                'mean': sum(sorted_vals) / n,
                'median': sorted_vals[n // 2] if n % 2 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2,
                'min': sorted_vals[0],
                'max': sorted_vals[-1]
            }
        
        bleu_scores = [m.bleu for m in metrics if m.bleu is not None]
        groundedness_scores = [m.groundedness for m in metrics if m.groundedness is not None]
        hallucination_rates = [m.hallucination_rate for m in metrics if m.hallucination_rate is not None]
        
        return {
            'count': len(metrics),
            'bleu': calc_stats(bleu_scores),
            'groundedness': calc_stats(groundedness_scores),
            'hallucination_rate': calc_stats(hallucination_rates)
        }
    
    def export_to_csv(self, output_path: str = "rag_metrics.csv"):
        """Export metrics to CSV format."""
        import csv
        
        metrics = self.load_metrics()
        if not metrics:
            print("No metrics to export.")
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'bleu', 'groundedness', 'hallucination_rate']
            
            # Add metadata fields
            if metrics[0].metadata:
                fieldnames.extend(metrics[0].metadata.keys())
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in metrics:
                row = {
                    'timestamp': m.timestamp,
                    'bleu': m.bleu,
                    'groundedness': m.groundedness,
                    'hallucination_rate': m.hallucination_rate
                }
                row.update(m.metadata)
                writer.writerow(row)
        
        print(f"Exported {len(metrics)} metrics to {output_path}")
