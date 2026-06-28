# Enterprise Evaluation Framework

This directory contains the evaluation framework for the NovaTech Enterprise Knowledge Assistant. It provides a structured methodology to measure the quality of the RAG pipeline across three key dimensions: Retrieval, Answer Quality, and Citation Accuracy.

## Evaluation Methodology

### 1. Retrieval Metrics (`retrieval_eval.py`)
- **Precision@K:** Measures the fraction of the top-K retrieved documents that are relevant to the query. Higher precision means less noise is sent to the LLM.
- **Recall@K:** Measures the fraction of all relevant documents that appear in the top-K. High recall ensures the LLM has all the necessary information.
- **Mean Reciprocal Rank (MRR):** Evaluates how high the first relevant document ranks in the results. Essential for ensuring the most important context is seen first.

### 2. Answer Quality Metrics (`answer_eval.py`)
- **Answer Relevance:** Checks if the generated answer actually addresses the user's question, utilizing expected keywords/phrases.
- **Groundedness:** Evaluates whether every claim in the answer is supported by the retrieved context, penalizing hallucinations.

### 3. Citation Metrics (`citation_eval.py`)
- **Citation Accuracy:** Verifies that the sources cited by the LLM were actually part of the retrieved context and contain the relevant information.
- **Citation Completeness:** Ensures that all necessary source documents are cited to support the answer.

## Running the Evaluation

1. Ensure the FastAPI server is running (`python main.py`).
2. Run the evaluation script:
   ```bash
   python evaluation/evaluate.py
   ```
3. The script will execute the queries defined in `test_cases.json` against the live API.
4. Results are compiled and saved to `evaluation_report.csv`.
