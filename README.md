# General-Purpose RAG Evaluation Harness

A pluggable evaluation framework for Retrieval-Augmented Generation (RAG) systems. Measure *any* retriever/generator pair on retrieval quality, answer hallucination, latency, and cost—without coupling evaluation logic to specific implementations.

## Problem

RAG systems are hard to evaluate reliably. Existing tools often:
- Bake assumptions about *how* retrieval/generation work (embedding model, LLM, chunking strategy)
- Make it painful to swap components or compare different architectures
- Lack comprehensive metrics (retrieval, generation quality, operational costs)

This harness solves that: **define what retrieval and generation *are* (as Protocols), build metrics that work with *any* implementation, plug systems in and compare.**

## Architecture

### Core Design: Protocol-Based Contracts

Instead of inheritance or concrete implementations, the harness uses Python `Protocol` (structural typing):

```python
class Retriever(Protocol):
    def retrieve(self, question: str) -> list[str]: ...

class Generator(Protocol):
    def generate(self, question: str, chunks: list[str]) -> str: ...
```

Any system with these method signatures works—no import of base classes, no coupling. This is why Study Buddy and a Mock system run through the same harness unchanged.

### Folder Structure

```
rag-eval-harness/
├── contracts/           # Protocol definitions (universal)
│   └── base.py         # Retriever, Generator contracts
├── adapters/           # Pluggable system implementations
│   ├── study_buddy.py  # Real RAG system (ChromaDB + Groq)
│   └── mock_system.py  # Proof-of-concept mock
├── harness/            # Orchestration
│   ├── runner.py       # Harness class (runs retriever→generator)
│   └── evaluate.py     # Batch evaluation, regression testing
├── metrics/            # Evaluation metrics
│   ├── retrieval.py    # recall_at_k, precision_at_k (embedding-based)
│   ├── hallucination.py # LLM-as-judge groundedness check
│   └── latency.py      # time_call (latency), calculate_cost
├── golden_dataset/     # Ground truth for evaluation
│   ├── os_module1.json # 5 OS questions with expected answers
│   └── loader.py       # Dataset loader
└── results/            # Timestamped evaluation runs
```

## Key Features

### 1. Retrieval Metrics

**Recall & Precision** — embedding-similarity based, using cosine distance. Accounts for paraphrasing (threshold calibrated to 0.6 for MiniLM embeddings on OS content).

```python
recall = recall_at_k(retrieved_chunks, golden_chunks, model, threshold=0.6)
precision = precision_at_k(retrieved_chunks, golden_chunks, model, threshold=0.6)
```

### 2. Hallucination Detection

**LLM-as-Judge** — sends answer + context to Groq, asks if answer is grounded. Returns structured JSON verdict + reasoning.

```python
verdict = judge_groundedness(answer, chunks, groq_client)
# Returns: {"verdict": "grounded"/"hallucinated", "reasoning": "..."}
```

### 3. Operational Metrics

**Latency & Cost** — generic `time_call` wrapper captures duration; token counts from LLM response tracked via side-channel (`self.last_usage`), converted to USD using Groq's pricing.

```python
cost = calculate_cost(usage_dict, input_price_per_million=0.15, output_price_per_million=0.60)
```

### 4. Regression Testing

**Batch Evaluation** — run all golden examples, save timestamped JSON results. Compare across runs after code changes (different embedding models, chunk sizes, prompts, etc.).

```python
eval_results = run_full_evaluation(retriever, generator, model, client, golden_data)
# Saved to results/eval_2026-08-24T16-15-40.json
```

## Design Decisions

### 1. Study Buddy Scope: Simplified Generator

Study Buddy's real implementation is an **agent** (tool-calling, decides whether to search notes vs. web). The harness uses a **simplified, non-agentic generator** that takes question + chunks and returns an answer. This is:
- Cleaner to evaluate (agent behavior is harder to measure)
- Explicitly documented as a scoping choice (not a limitation)
- Kept separate from the harness (harness doesn't know it's simplified)

### 2. Cost Tracking Without Breaking Contracts

Generator's return type is `-> str` (per Protocol). Storing token usage as a side-channel (`self.last_usage`) instead of changing the return type preserves the contract and lets code outside call `generator.last_usage` after generation.

### 3. Threshold Calibration

Recall/precision threshold (0.6) was empirically calibrated by running on real Study Buddy + MiniLM embeddings against OS notes. Not universal—worth revisiting if swapping embedding models.

## Running It

### Setup

```bash
# Install dependencies
pip install chromadb sentence-transformers langchain-groq python-dotenv

# Set Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Ingest notes into ChromaDB
python -c "from adapters.study_buddy import StudyBuddyRetriever, process_pdf; ...; process_pdf('notes.pdf', model, collection)"
```

### Evaluate Study Buddy

```python
from adapters.study_buddy import StudyBuddyRetriever, StudyBuddyGenerator
from harness.evaluate import run_full_evaluation

retriever = StudyBuddyRetriever(model, collection)
generator = StudyBuddyGenerator(client)
results = run_full_evaluation(retriever, generator, model, client, golden_data)
# Saved to results/eval_*.json
```

### Prove Generality (Mock System)

```python
from adapters.mock_system import MockRetriever, MockGenerator
from harness.evaluate import run_full_evaluation

mock_retriever = MockRetriever()
mock_generator = MockGenerator()
results = run_full_evaluation(mock_retriever, mock_generator, model, client, golden_data)
# Same harness, no changes. Works with anything that matches Protocol.
```

## Metrics Explained

| Metric | What It Measures | Range | Interpretation |
|--------|------------------|-------|-----------------|
| **Recall** | % of golden chunks found by retriever | 0–1 | Higher = better retrieval coverage |
| **Precision** | % of retrieved chunks actually relevant | 0–1 | Higher = less noise in retrieval |
| **Hallucination** | Verdict on answer groundedness | {grounded, hallucinated} | grounded = safe answer |
| **Latency** | Seconds for retrieval + generation | 0–∞ | Lower = faster UX |
| **Cost** | USD spent on API calls | 0–∞ | Lower = cheaper to run at scale |

## Future Work

- **Multi-metric comparison** — diff results across Study Buddy → Mock, or before/after prompt changes
- **Precision** — add `precision_at_k` (currently stub, ready to implement)
- **Dashboard** — simple web UI showing evaluation trends over time
- **Phase 7+ adapters** — plug in other retrieval strategies (BM25, keyword, hybrid) to compare against embeddings

## Technology Stack

- **Python 3.10+** — core language
- **ChromaDB** — vector database (Study Buddy retrieval)
- **SentenceTransformer** — embedding model (MiniLM-L6-v2)
- **Groq API** — fast LLM inference (llama-3.1-8b, gpt-oss-120b)
- **LangChain** — LLM orchestration
- **Typing (Protocol)** — structural typing for contracts

## Why This Matters

This harness proves you can build evaluation infrastructure that's **decoupled from specific systems**. It's the kind of thinking required for:
- **Platform/SDK design** — define contracts, let others implement
- **A/B testing at scale** — swap components, measure impact
- **Research** — compare completely different RAG approaches (embedding vs. BM25 vs. hybrid) fairly

## Author

Built as a portfolio project to demonstrate:
- Full-stack system design (contracts → metrics → orchestration)
- Working with real APIs (Groq, ChromaDB)
- Deliberate engineering decisions with documented tradeoffs
- End-to-end testing and debugging