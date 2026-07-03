# RAG for Financial Question Answering on SEC 10-K Filings

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about
company annual reports (SEC 10-K filings), evaluated against a standalone LLM
baseline that uses the **same generator model** but no retrieval.

> **MSc dissertation (COMP702), University of Liverpool, 2025/26.**
> Supervisor: Dr. Blaine Keetch.

## Research question

Does retrieval-augmented generation improve the accuracy and faithfulness of an
LLM answering questions about financial filings, compared to the same LLM
answering from its own parametric knowledge alone?

To answer this fairly, both systems share an identical generator (GPT-4o-mini)
and differ in only one variable — whether relevant filing text is retrieved and
supplied as context. Any measured difference is therefore attributable to
retrieval.

## Architecture

The system has two pipelines.

### 1. Offline indexing (run once per company)

```
SEC EDGAR (HTML 10-K)
  → requests + BeautifulSoup (parse to text)
  → RecursiveCharacterTextSplitter (512-token chunks, 50 overlap)
  → text-embedding-3-small (embeddings)
  → Chroma vector store (persisted to disk, cosine similarity)
```

### 2. Online query (run per question, for both systems)

```
Question
  ├─ RAG:      embed → top-k=5 cosine retrieval from Chroma → GPT-4o-mini with context
  └─ Baseline: GPT-4o-mini with no context (same model, no retrieval)
```

## Project structure

```
.
├── src/
│   ├── ingest.py    # Download a 10-K from SEC EDGAR and save it as text
│   ├── index.py     # Chunk, embed, and store the text in Chroma
│   └── query.py     # Answer a question via RAG and via the baseline
├── data/            # Downloaded filings (gitignored, rebuilt by ingest.py)
├── chroma_db/       # Persisted vector store (gitignored, rebuilt by index.py)
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+ and an OpenAI API key.

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide your OpenAI API key in a .env file at the project root
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

Run all commands **from the project root** (the scripts use paths relative to it).

```bash
# 1. Download a company's 10-K (currently configured for Apple, FY2023)
python3 src/ingest.py

# 2. Build the vector store from the downloaded filing
python3 src/index.py

# 3. Ask a question — prints a RAG answer, a baseline answer, and the retrieved context
python3 src/query.py "What was Apple's total net sales in fiscal 2023?"
```

`query.py` also runs interactively if you omit the question:

```bash
python3 src/query.py
# Question: <type your question>
```

## Evaluation

- **Primary:** RAGAS metrics — Faithfulness, Answer Relevance, Context Relevance,
  Context Recall.
- **Secondary:** Exact Match and F1 against 50 manually constructed QA pairs
  (factual, numerical, and risk-based questions).

## Dataset

FY2023 10-K filings for five companies spanning four sectors:

| Ticker | Company           | Sector     |
|--------|-------------------|------------|
| AAPL   | Apple             | Technology |
| MSFT   | Microsoft         | Technology |
| JPM    | JPMorgan Chase    | Banking    |
| XOM    | ExxonMobil        | Energy     |
| JNJ    | Johnson & Johnson | Healthcare |

Filings are downloaded directly from [SEC EDGAR](https://www.sec.gov/edgar).
EDGAR requires a descriptive `User-Agent` header on all requests.

## Key design decisions

- **Chunk size 512 / overlap 50** — matches the embedding model's effective range
  and reduces "lost in the middle" risk.
- **k = 5 retrieval** — keeps context compact to avoid long-context degradation.
- **Dense retrieval (embeddings) over sparse (BM25)** — stronger on
  knowledge-intensive QA.
- **Identical generator (GPT-4o-mini) for both systems** — isolates retrieval as
  the only variable; `temperature=0` for reproducibility.
- **Naive RAG** — advanced/modular RAG techniques are out of scope for the core
  system.

## Status

- [x] Offline indexing pipeline (Apple, FY2023)
- [x] Online query pipeline (RAG + baseline)
- [ ] 50 QA pairs
- [ ] RAGAS + Exact Match / F1 evaluation
- [ ] Scale ingestion and indexing to all five companies

## Known limitations

Plain-text chunking handles tabular and numerical data poorly — table structure
is lost when HTML is flattened to text. This is a documented, expected finding of
the study rather than a defect to be fixed.
