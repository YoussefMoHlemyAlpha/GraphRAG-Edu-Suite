# 🎓 GraphRAG Edu-Suite with Quality Control

> Intelligent Educational Content Generation with Comprehensive Quality Assurance

Generate high-quality, context-grounded quiz questions and essay prompts using Graph-based RAG, dual-model AI validation, and comprehensive quality metrics.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Neo4j](https://img.shields.io/badge/neo4j-5.0+-green.svg)](https://neo4j.com/)

---

## 🌟 Key Features

### 🎯 Quality-Controlled Question Generation
- **Multiple Choice Questions (MCQs)**: 6-10 questions with Bloom's Taxonomy coverage
- **Essay Questions**: 2-5 deep-reasoning prompts with expected concepts
- **Per-Question Metrics**: Individual quality scores for each question
- **Automatic Filtering**: Low-quality questions rejected automatically (>50% coverage threshold)
- **Answer Verification**: Validates correct answers for MCQs
- **Concept Verification**: Checks expected concepts for essays

### 📊 Comprehensive RAG Metrics
- **Groundedness**: Measures how well content is supported by source material (target: >90%)
- **Hallucination Rate**: Tracks content not found in retrieved context (target: <25%)
- **Overall Quality**: Combined score (target: >80% for "Excellent")
- **Real-time Tracking**: Monitor quality across all generated content

### 🔍 Advanced Validation System
- **Dual-Model Architecture**: Gemma3:4b for generation, Llama 3.2 for validation
- **Coverage Checks**: Questions must have >50% term overlap with context
- **Critic Validation**: 5-point validation checklist for every question
- **Fuzzy Matching**: Reduces false positives for technical terms

### 📈 Proven Quality Improvements
- **50% reduction** in hallucination rate (50.6% → 20-25%)
- **100% fix** for wrong answers (2/6 → 0/6)
- **11% increase** in overall quality (74.7% → 83-87%)
- **100% question generation** (3/6 → 6/6)

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Neo4j Database** (running locally or remotely)
3. **Ollama** with required models

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GraphRAG-Edu-Suite.git
cd GraphRAG-Edu-Suite

# Install dependencies
pip install -r requirements.txt

# Pull required AI models
ollama pull gemma3:4b
ollama pull llama3.2:latest

# Configure environment
cp .env.example .env
# Edit .env with your Neo4j credentials:
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your_password
```

### Run the Application

```bash
streamlit run main.py
```

Visit `http://localhost:8501` in your browser!

---

## 📖 How It Works

### 1️⃣ Upload Content
Upload PDF lecture materials to build your knowledge graph. The system extracts entities, relationships, and concepts using Gemma3.

### 2️⃣ Generate Questions
Choose between MCQs or Essays, select your lesson, and specify the number of questions.

### 3️⃣ Automatic Quality Validation
The system automatically:
- Retrieves 1500+ fys, select lesson and question count

### 3. Quality Validation
System automatically:
- Retrieves 1500+ facts from knowledge graph
- Generates questions with Gemma3
- Validates with Llama 3.2 critic
- Filters by coverage thresholds
- Calculates RAG metrics

### 4. Review & Use
See per-question quality metrics and use high-quality content!

---

## 📊 Example Output

### MCQ with Quality Metrics:
```
Q1: Understand
Which of the following best describes Data-Driven AI?

Groundedness: 100% ✅ Good
Hallucination: 20% ✅ Low

A. An AI approach that learns from examples and data. ✓
B. A system that relies solely on human-defined rules.
C. A system that mimics human expert decision-making.
D. A system using symbolic reasoning.
```

### Overall Quality:
```
📊 Overall Quiz Quality
Average Groundedness: 100%
Average Hallucination: 🟢 24%
Overall Quality: 88% ⭐ Excellent
```

---

## 🏗️ Architecture

```
User Interface (Streamlit)
    ↓
Generation Pipeline
    ├─ Context Retrieval (Neo4j)
    ├─ Generation (Gemma3:4b)
    ├─ Validation (Llama 3.2)
    ├─ Coverage Filtering
    └─ RAG Metrics
    ↓
Knowledge Graph (Neo4j)
```

---

## 📁 Project Structure

```
GenertaiveQestionsModel/
├── main.py                 # Streamlit UI
├── engine/
│   ├── processor.py        # PDF processing
│   ├── generator.py        # Question generation
│   ├── graph_store.py      # Neo4j operations
│   ├── rag_metrics.py      # Metrics calculation
│   └── vram_util.py        # Model management
├── docs/                   # Documentation
├── .env                    # Configuration
└── requirements.txt        # Dependencies
```

---

## 🎯 Quality Metrics Explained

### Groundedness
Measures how well questions are supported by source material
- **Formula**: (Supported Sentences) / (Total Sentences)
- **Target**: >90%

### Hallucination Rate
Tracks content not found in retrieved context
- **Formula**: (Words NOT in Context) / (Total Unique Words)
- **Target**: <25%

### Overall Quality
Combined quality score
- **Formula**: (Groundedness × 50%) + ((1 - Hallucination) × 50%)
- **Target**: >80% for "Excellent"

---

## 📚 Documentation

- [Project Overview](docs/PROJECT_OVERVIEW_V2.md) - Complete system documentation
- [Metrics Guide](docs/METRICS_CALCULATION_EXPLAINED.md) - How metrics are calculated
- [Visual Guide](docs/METRICS_VISUAL_GUIDE.md) - Visual explanations
- [Testing Guide](docs/TESTING_IMPROVEMENTS.md) - How to test improvements
- [Before/After](docs/BEFORE_AFTER_COMPARISON.md) - Quality improvements comparison

---

## 🔧 Configuration

### Environment Variables (.env)
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### Model Configuration
- **Gemma3:4b**: Generation model (questions, essays)
- **Llama 3.2**: Validation model (critic, grading)

### Quality Thresholds
- Question coverage: >50% (configurable in `engine/generator.py`)
- Answer coverage: >40% (configurable in `engine/generator.py`)
- Concept coverage: >40% (configurable in `engine/generator.py`)

---

## 📈 Performance

### Before Quality Control:
- Hallucination: 50.6%
- Wrong answers: 2/6 questions
- Question count: 3/6 generated

### After Quality Control:
- Hallucination: 20-25% ✅
- Wrong answers: 0/6 questions ✅
- Question count: 6/6 generated ✅

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional LLM support
- Enhanced metrics algorithms
- UI/UX improvements
- Documentation
- Bug fixes


---

## 🙏 Acknowledgments

- Ollama for local LLM inference
- Neo4j for graph database
- Streamlit for web framework
- LangChain for LLM orchestration


---

## 🎓 Use Cases

### For Educators:
- Generate quiz questions from lectures
- Create essay prompts with quality assurance
- Track content quality
- Ensure questions are answerable

### For Students:
- Practice with high-quality questions
- Get clear essay expectations
- Receive AI-powered feedback
- Study with Bloom's Taxonomy alignment

### For Researchers:
- Experiment with RAG metrics
- Study hallucination reduction
- Analyze knowledge graphs
- Benchmark LLM performance

---

**Version**: 2.0 (Quality Control Update)  
**Status**: Production Ready ✅  
**Last Updated**: February 2026

---

Made with ❤️ using Gemma3, Llama 3.2, Neo4j, and Streamlit
