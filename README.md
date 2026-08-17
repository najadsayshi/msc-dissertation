# RAG for Financial Question Answering on SEC 10-K Filings

A Retrieval-Augmented Generation (RAG) pipeline for answering questions about
company annual reports (SEC 10-K filings), evaluated against a standalone LLM
baseline that uses the **same generator model** but no retrieval.

> **MSc dissertation (COMP702), University of Liverpool, 2024/25.**
> Supervisor: Dr. Blaine Keetch.

## Research question

Does retrieval-augmented generation improve the accuracy and faithfulness of an
LLM answering questions about financial filings, compared to the same LLM
answering from its own parametric knowledge alone?

Both systems share an identical generator (GPT-4o-mini) and differ in only one
variable — whether relevant filing text is retrieved and supplied as context. Any
measured difference is therefore attributable to retrieval.

## Architecture

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
│   ├── query.py     # Answer a question via RAG and via the baseline
│   ├── evaluate.py  # Score both systems over the QA set
│   └── app.py       # Web UI comparing the two systems side by side
├── data/
│   └── qa_pairs.json  # QA benchmark (tracked; the filings are gitignored)
├── chroma_db/       # Vector store (gitignored, rebuilt by index.py)
├── results/         # Evaluation outputs (gitignored, rebuilt by evaluate.py)
├── requirements.txt
└── README.md
```

`data/qa_pairs.json` is the only file here that cannot be regenerated, which is why
`.gitignore` ignores `data/*` with a single exception for it.

## Setup

Requires Python 3.10+ and an OpenAI API key.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

Run all commands **from the project root** — the scripts use paths relative to it.

```bash
# 1. Download the filings (--ticker AAPL|MSFT|JPM|XOM|JNJ, or --all)
python3 src/ingest.py --all

# 2. Build the vector store (safe to re-run)
python3 src/index.py --all

# 3. Ask a question — prints the RAG answer, the baseline answer, and the context
python3 src/query.py "What was Apple's total net sales in fiscal 2023?"

# 4. Score both systems over the QA benchmark
python3 src/evaluate.py --run-name all
python3 src/evaluate.py --company AAPL --run-name apple   # one company only
```

Answers are written to `results/<run-name>_records.jsonl` as each question
completes, so an interrupted run resumes rather than paying for the same API calls
twice. After editing a QA pair, delete that file — the cache is keyed on question
ID and would otherwise reuse a stale answer.

`query.py` also runs interactively if you omit the question.

### Comparison UI

```bash
uvicorn app:app --app-dir src --reload
```

Then open `http://127.0.0.1:8000`. Pick a benchmark question or type your own, and
both systems answer side by side with the retrieved passages shown beneath — each
labelled with the filing it came from, so cross-company retrieval is visible.

RAGAS scores for benchmark questions are read from the frozen evaluation run, so
they appear instantly and match the reported results. A **Score this live** button
re-runs RAGAS on a single question on demand; that takes a couple of minutes,
because faithfulness and context recall each make several judge calls. A second tab
shows the aggregate results and the scoring coverage.

## Dataset

FY2023 10-K filings for five companies spanning four sectors, downloaded from
[SEC EDGAR](https://www.sec.gov/edgar), which requires a descriptive `User-Agent`
header on all requests.

| Ticker | Company           | Sector     |
|--------|-------------------|------------|
| AAPL   | Apple             | Technology |
| MSFT   | Microsoft         | Technology |
| JPM    | JPMorgan Chase    | Banking    |
| XOM    | ExxonMobil        | Energy     |
| JNJ    | Johnson & Johnson | Healthcare |

The benchmark is 50 hand-authored question–answer pairs, 10 per company, split
across numerical, factual and risk-based questions. No existing benchmark covers
these companies for this period. Each pair records the sentence from the filing
that contains the answer, so ground truths can be checked against the source.

Questions are deliberately worded away from the filings' own phrasing. A question
that copies the sentence answering it lets retrieval succeed on word overlap alone,
which measures string matching rather than semantic search.

## Evaluation

- **Primary:** RAGAS — Faithfulness, Answer Relevance, Context Relevance and
  Context Recall, judged by `gpt-4o`.
- **Secondary:** Exact Match and F1 against the QA pairs.

Every RAGAS mean is reported with the number of questions actually scored beside
it. The judge can fail on individual questions and return no score, and an average
over part of the set is not the same claim as an average over all of it.

Retrieval is **not** filtered by ticker: all five filings share one collection and
the system is never told which company a question refers to. Cross-company
retrieval errors are measured rather than prevented, and the evaluation reports how
many of the five retrieved chunks came from the correct company as evidence for
that choice.

## Key design decisions

- **Chunk size 512 / overlap 50** — matches the embedding model's effective range
  and reduces "lost in the middle" risk.
- **k = 5 retrieval** — keeps context compact to avoid long-context degradation.
- **Dense retrieval over BM25** — stronger on knowledge-intensive QA.
- **Identical generator for both systems** — isolates retrieval as the only
  variable, with `temperature=0` for reproducibility. The two system prompts differ
  *only* in the presence of context.
- **`gpt-4o` as the judge** — stronger than either system under test, so the
  generator is not marking its own homework.
- **No ticker filter** — a real user does not say which of five documents to
  search, so neither does the evaluation.
- **Naive RAG** — advanced and modular RAG techniques are out of scope.

## Status

- [x] Offline indexing pipeline, all five companies
- [x] Online query pipeline (RAG + baseline)
- [x] Evaluation harness with checkpointing
- [x] QA benchmark at 50 pairs
- [x] RAGAS scoring coverage reported alongside every mean
- [ ] Verification pass over the QA pairs
- [ ] Final run with pinned model snapshots
- [ ] Frozen results committed to a dated file

## Known limitations

Documented findings of the study rather than defects to be fixed.

- **Plain-text chunking handles tables poorly.** Flattening HTML puts every table
  cell on its own line, separating a figure from the header that gives it scale.
- **Exact Match understates the system.** It requires exact string equality, so a
  correct answer phrased differently from the ground truth scores zero. This is why
  RAGAS is the primary metric.
- **String metrics are weakest on risk questions**, where a correct answer
  paraphrases rather than reproducing the ground truth.
- **Answer relevance is not accuracy.** A confidently wrong answer can score well
  on it, and a terse correct one can score badly.
- **GPT-4o-mini's training cutoff predates these filings.** Rather than a confound,
  this is the point: it is exactly the situation in which retrieval should help.
