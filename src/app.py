# app.py - side-by-side comparison UI: RAG vs the same model with no retrieval.
# no pipeline logic here. it calls the exact functions evaluate.py scores, so what
# the page shows is the system that was measured.

import os
import sys
import json

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from query import answer_rag, answer_baseline, CHAT_MODEL, TOP_K
from ingest import COMPANIES

QA_FILE = "data/qa_pairs.json"
RESULTS_FILE = "results/final_results.json"
SUMMARY_FILE = "results/final_summary.json"

# front end lives in src/static - resolved off this file so it doesn't matter
# which directory uvicorn was launched from
STATIC_DIR = os.path.join(HERE, "static")

app = FastAPI(title="RAG vs Baseline - SEC 10-K QA")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# saved scores from the frozen evaluation run, keyed by question id
def saved_scores() -> dict:
    records = read_json(RESULTS_FILE)
    if not records:
        return {}

    scores = {}
    for r in records:
        scores[r["id"]] = {
            "rag_ragas": r.get("rag_ragas", {}),
            "baseline_ragas": r.get("baseline_ragas", {}),
            "rag_em": r.get("rag_em"),
            "rag_f1": r.get("rag_f1"),
            "baseline_em": r.get("baseline_em"),
            "baseline_f1": r.get("baseline_f1"),
        }
    return scores


class AskRequest(BaseModel):
    question: str


class ScoreRequest(BaseModel):
    question: str
    rag_answer: str
    baseline_answer: str
    contexts: list
    ground_truth: str = ""


@app.get("/api/companies")
def api_companies():
    out = []
    for ticker in sorted(COMPANIES.keys()):
        out.append({"ticker": ticker, "name": COMPANIES[ticker]["name"]})
    return out


@app.get("/api/questions")
def api_questions():
    # benchmark questions plus whatever the frozen run scored them
    qa_pairs = read_json(QA_FILE) or []
    scores = saved_scores()

    out = []
    for qa in qa_pairs:
        item = dict(qa)
        item["saved"] = scores.get(qa["id"])
        out.append(item)
    return out


@app.post("/api/ask")
def api_ask(req: AskRequest):
    # the actual experiment: same model twice, only retrieval differs
    rag = answer_rag(req.question)
    baseline = answer_baseline(req.question)

    return {
        "rag_answer": rag["answer"],
        "baseline_answer": baseline,
        "contexts": rag["contexts"],
        "tickers": rag["tickers"],
    }


@app.post("/api/score")
def api_score(req: ScoreRequest):
    # score ONE question live. run_ragas already takes a list, so a list of one
    # works as-is. imported here so the page still loads if ragas is unhappy.
    from evaluate import run_ragas

    record = {
        "id": "live",
        "question": req.question,
        "ground_truth": req.ground_truth,
        "rag_answer": req.rag_answer,
        "baseline_answer": req.baseline_answer,
        "contexts": req.contexts,
    }

    # no ground truth -> context recall/precision have nothing to compare against
    if req.ground_truth.strip():
        note = ""
    else:
        note = "No ground truth for a custom question, so context metrics cannot be computed."

    run_ragas([record])

    return {
        "rag_ragas": record.get("rag_ragas", {}),
        "baseline_ragas": record.get("baseline_ragas", {}),
        "note": note,
    }


@app.get("/api/summary")
def api_summary():
    summary = read_json(SUMMARY_FILE)
    if not summary:
        return {"available": False}
    return {"available": True, "summary": summary}


@app.get("/api/config")
def api_config():
    return {"chat_model": CHAT_MODEL, "top_k": TOP_K}


@app.get("/")
def index():
    # the page itself lives in src/static/index.html
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
