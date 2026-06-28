# System Design Document

## 1. High-Level Architecture

The Enterprise Knowledge Assistant uses a modern **Retrieval Augmented Generation (RAG)** pipeline designed for accuracy, transparency, and safety.

```
┌────────────────────────────────────────────────────────────┐
│                    Web Frontend (SPA)                       │
│  Chat UI · Document Upload · Source Citations · Feedback    │
└────────────────────────┬───────────────────────────────────┘
                         │ REST API (FastAPI)
┌────────────────────────▼───────────────────────────────────┐
│              RAG Pipeline (Orchestrator)                    │
│                                                            │
│  1. User Query + Session ID                                │
│       ↓                                                    │
│  2. Query Rewriter (Gemini: Contextual resolution)         │
│       ↓                                                    │
│  3. Hybrid Retriever (Semantic + BM25)                     │
│       ↓                                                    │
│  4. Reciprocal Rank Fusion (RRF)                           │
│       ↓                                                    │
│  5. Cross-Encoder Re-ranker (MS-MARCO)                     │
│       ↓                                                    │
│  6. Context Validation Guardrails (Safety Layer)           │
│       ↓                                                    │
│  7. LLM Generation (Gemini 2.0 Flash)                      │
│       ↓                                                    │
│  8. Confidence Engine & Response Formatting                │
│                                                            │
├──────────┬──────────────┬─────────────────────────────────┤
│ Embedding│  ChromaDB    │   BM25 Index   │  Gemini LLM    │
│ Model    │  (Vectors)   │   (Keywords)   │  (Generation)  │
└──────────┴──────────────┴────────────────┴────────────────┘
```

## 2. Data Flow & Component Explanation

### 2.1 Ingestion Phase
1. **Document Processor:** Extracts text from files (PDF/DOCX/MD/TXT) preserving page metadata.
2. **Chunking:** Applies recursive character splitting (~500 chars) with overlap (~50 chars) while respecting paragraph boundaries.
3. **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional dense vectors.
4. **Storage:** Vectors and metadata are stored in **ChromaDB**. Raw text is tokenized and added to the **BM25 Index**.

### 2.2 Query Phase
1. **Query Rewriting:** Uses the LLM and `ConversationMemory` to resolve pronouns (e.g., "What is it?" → "What is the leave policy?").
2. **Hybrid Retrieval:**
   - *Semantic Search:* Fetches top-10 chunks via ChromaDB cosine similarity.
   - *Keyword Search:* Fetches top-10 chunks via BM25 (critical for acronyms and product names).
3. **Merging:** Reciprocal Rank Fusion (RRF) merges the two lists.
4. **Re-ranking:** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores query-chunk pairs and sorts the top-5 candidates for maximum precision.

### 2.3 Generation & Safety Phase
1. **Context Validation (Guardrails):** Evaluates the top chunks. If the best score is below `0.35`, the system blocks LLM generation and returns a safe refusal message. This prevents hallucinations when the knowledge base lacks relevant data.
2. **LLM Generation:** The top chunks are injected into a strict system prompt instructing Gemini to answer *only* from the context and to provide citations.
3. **Confidence Scoring:** Calculates `0.6 * retrieval_score + 0.4 * reranker_score` to assign a High/Medium/Low badge to the UI.
4. **Formatting:** Extracts 150-character excerpts directly from the chunks to display alongside document/page citations.

## 3. Design Decisions

### Why Hybrid Search + Re-ranking?
Vector search alone struggles with exact keyword matches (e.g., "Error Code 404"). BM25 catches keywords but misses semantic intent. Combining both via RRF provides the best candidate pool. The cross-encoder is computationally heavy, so it is applied *only* to the top-10 merged candidates to refine the final top-5 context injected into the LLM.

### Hallucination Prevention Strategy
LLMs (even GPT-4/Gemini) are prone to "sycophancy" — they want to answer questions even when context is weak. Instead of relying purely on prompting ("Don't answer if you don't know"), we built a hard **Guardrail Layer** (`app/guardrails.py`). By gating generation behind a retrieval score threshold, we physically prevent the LLM from hallucinating on out-of-scope queries.

### Enhanced Citations with Excerpts
Returning `{"doc": "Policy.pdf", "page": 2}` requires the user to open the PDF to verify the AI's claim. By including a direct 150-character excerpt extracted directly from the semantic chunk, we enable instant verification in the UI, dramatically increasing enterprise trust.

## 4. Scalability Considerations

Currently, the app uses in-memory SQLite (ChromaDB) and pickle (BM25), which is perfect for an assignment demo. To scale to millions of documents:

1. **Database:** Replace ChromaDB with managed Pinecone or self-hosted Qdrant/Milvus. Replace pickled BM25 with Elasticsearch.
2. **Asynchronous Ingestion:** Decouple document upload from processing using an event queue (RabbitMQ/Kafka) and Celery workers.
3. **LLM Scalability:** Gemini API easily scales, but a fallback/load-balancing strategy (e.g., litellm) could route traffic between Gemini, OpenAI, and Anthropic.
4. **Frontend:** Extract the frontend into a separate React/Next.js application hosted on Vercel/S3.
