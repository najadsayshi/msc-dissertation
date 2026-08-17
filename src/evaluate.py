# evaluate.py - scores the answers.

import os
import re
import json
import string
import argparse
import collections

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,   
    context_recall,
)

# importing the actual pipeline from query.py 
from query import answer_rag, answer_baseline, EMBEDDING_MODEL, TEMPERATURE

load_dotenv()

QA_FILE = "data/qa_pairs.json"
RESULTS_DIR = "results"

# the marker or the judge
JUDGE_MODEL = "gpt-4o"

# which RAGAS metrics each system gets
RAG_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
BASELINE_METRICS = [answer_relevancy]

# ragas defaults to 16 questions judged at once, and faithfulness/context_recall
# each fan out into several calls per question. that swamped the rate limit and
# 30 of 50 came back NaN. 4 is slower and actually finishes.
JUDGE_RUN_CONFIG = RunConfig(max_workers=4, timeout=300, max_retries=10)


# String checks: EM and F1. No API calls, no cost.

# EDGAR text is full of typographic quotes and dashes. string.punctuation only
# knows the ASCII ones, so "Company's" and "Company's" wouldn't match otherwise
SMART_CHARS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
}


def normalize(text: str) -> str:
    # lowercase, strip punctuation, drop a/an/the, squash spaces
    text = text.lower()

    for smart, plain in SMART_CHARS.items():
        text = text.replace(smart, plain)

    kept_chars = []
    for ch in text:
        if ch not in string.punctuation:
            kept_chars.append(ch)
    text = "".join(kept_chars)

    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize(prediction) == normalize(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    # word overlap between answer and truth, 0 to 1
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        # empty only matches empty
        if pred_tokens == gt_tokens:
            return 1.0
        else:
            return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)   # shared words / what the model said
    recall = overlap / len(gt_tokens)        # shared words / what the truth was
    return 2 * precision * recall / (precision + recall)


# not a score - a diagnostic for the no-ticker-filter decision. how many of the
# 5 chunks came from the company the question is actually about
def own_company_chunks(tickers: list, company: str) -> int:
    count = 0
    for ticker in tickers:
        if ticker == company:
            count += 1
    return count


# Step 1 - ask both systems everything
def score_pair(qa: dict) -> dict:
    question = qa["question"]
    gt = qa["ground_truth"]

    rag = answer_rag(question)          # answer + the chunks it used
    baseline = answer_baseline(question)

    # one dict per question. EM/F1 are free, so compute them now
    return {
        "id": qa["id"],
        "type": qa["type"],
        "company": qa["company"],
        "question": question,
        "ground_truth": gt,
        "rag_answer": rag["answer"],
        "contexts": rag["contexts"],
        "baseline_answer": baseline,
        "rag_em": exact_match(rag["answer"], gt),
        "rag_f1": f1_score(rag["answer"], gt),
        "own_company_chunks": own_company_chunks(rag["tickers"], qa["company"]),
        "baseline_em": exact_match(baseline, gt),
        "baseline_f1": f1_score(baseline, gt),
    }


# rewrite the checkpoint with the judge's scores in it. answers were already
# safe across runs; the judging is the expensive half and wasn't
def save_checkpoint(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_checkpoint(path: str) -> dict:
    # whatever survived a previous run, keyed by question id
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done[record["id"]] = record
    return done


def run_systems(qa_pairs: list[dict], checkpoint_path: str) -> list[dict]:
    # answers get appended to disk as they finish - a crash at question 47 of 50
    # shouldn't mean paying for the first 46 all over again
    done = load_checkpoint(checkpoint_path)
    if done:
        print(f"Resuming: {len(done)} answers already on disk in {checkpoint_path}.\n")

    records = []
    for i, qa in enumerate(qa_pairs, 1):
        if qa["id"] in done:
            print(f"[{i}/{len(qa_pairs)}] {qa['id']}  (cached)")
            records.append(done[qa["id"]])
            continue

        print(f"[{i}/{len(qa_pairs)}] {qa['id']}  {qa['question']}")
        record = score_pair(qa)

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        records.append(record)

    return records


#  #
# Step 2 - RAGAS grading

def build_judge():
    # the LLM + embeddings RAGAS marks with
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=TEMPERATURE))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    return llm, embeddings


def run_ragas(records: list[dict]) -> None:
    llm, embeddings = build_judge()

    # RAG has context -> all four metrics
    # only judge what hasn't been judged. a re-run to patch 3 missing scores
    # shouldn't re-score the 47 that were already fine
    todo = []
    for r in records:
        if needs_judging(r):
            todo.append(r)

    skipped = len(records) - len(todo)
    if skipped:
        print(f"\n{skipped} of {len(records)} already fully judged - skipping those.")
    if not todo:
        print("Nothing left to judge.")
        return

    rag_samples = []
    for r in todo:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["rag_answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        rag_samples.append(sample)
    print(f"\nScoring RAG with RAGAS (4 metrics) on {len(todo)} questions...")
    rag_df = evaluate(
        dataset=EvaluationDataset(samples=rag_samples),
        metrics=RAG_METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=JUDGE_RUN_CONFIG,
    ).to_pandas()

    print(rag_df)

    # baseline -> Answer Relevance only
    base_samples = []
    for r in todo:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["baseline_answer"],
        )
        base_samples.append(sample)
    print("Scoring Baseline with RAGAS (Answer Relevance only)...")
    base_df = evaluate(
        dataset=EvaluationDataset(samples=base_samples),
        metrics=BASELINE_METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=JUDGE_RUN_CONFIG,
    ).to_pandas()

    # RAGAS keeps the rows in the order they were sent, so row i = record i.
    # a retry can still come back empty - keep whatever score already existed
    # rather than overwriting a good one with None
    for i, r in enumerate(todo):
        rag_scores = r.get("rag_ragas", {})
        for m in RAG_METRICS:
            fresh = _safe_float(rag_df.iloc[i][m.name])
            if fresh is not None or m.name not in rag_scores:
                rag_scores[m.name] = fresh
        r["rag_ragas"] = rag_scores

        baseline_scores = r.get("baseline_ragas", {})
        fresh = _safe_float(base_df.iloc[i]["answer_relevancy"])
        if fresh is not None or "answer_relevancy" not in baseline_scores:
            baseline_scores["answer_relevancy"] = fresh
        r["baseline_ragas"] = baseline_scores

    warn_if_incomplete(records)


# a record is done when every metric has a real score. anything still None means
# the judge failed on it and it's worth another go
def needs_judging(record: dict) -> bool:
    rag = record.get("rag_ragas")
    base = record.get("baseline_ragas")
    if not rag or not base:
        return True

    for m in RAG_METRICS:
        if not isinstance(rag.get(m.name), (int, float)):
            return True
    if not isinstance(base.get("answer_relevancy"), (int, float)):
        return True
    return False


# say so loudly when the judge didn't score everything - the means below are
# computed over whatever survived, which is easy to mistake for the full set
def warn_if_incomplete(records: list[dict]) -> None:
    short = []
    for m in RAG_METRICS:
        scored = 0
        for r in records:
            if isinstance(r["rag_ragas"].get(m.name), (int, float)):
                scored += 1
        if scored < len(records):
            short.append(f"{m.name} {scored}/{len(records)}")

    if short:
        print()
        print("!" * 70)
        print("WARNING: RAGAS did not score every question. Means below are over")
        print("the scored subset only, NOT all", len(records), "questions.")
        print("  " + "   ".join(short))
        print("!" * 70)


def _safe_float(value) -> float | None:
    # RAGAS returns NaN when it fails to score something - turn that into
    # None so the averages don't break
    try:
        f = float(value)
        # NaN is never equal to itself
        if f != f:
            return None
        else:
            return round(f, 3)
    except (TypeError, ValueError):
        return None


# Step 3 - average, print, save
def _mean(values: list) -> float | None:
    # average, skipping Nones
    nums = []
    for v in values:
        if isinstance(v, (int, float)):
            nums.append(v)

    if nums:
        return round(sum(nums) / len(nums), 3)
    else:
        return None


def _coverage(values: list) -> str:
    # how many of these actually got a score. RAGAS quietly returns NaN when a
    # judge call fails, and _mean skips those - without this a half-scored run
    # prints exactly like a full one
    scored = 0
    for v in values:
        if isinstance(v, (int, float)):
            scored += 1
    return f"{scored}/{len(values)}"


def agg(rows: list[dict]) -> dict:
    # one list per metric, then average each
    rag_faithfulness = []
    rag_answer_relevancy = []
    rag_context_precision = []
    rag_context_recall = []
    rag_em = []
    rag_f1 = []
    own_chunks = []
    baseline_answer_relevancy = []
    baseline_em = []
    baseline_f1 = []

    for r in rows:
        rag_faithfulness.append(r["rag_ragas"]["faithfulness"])
        rag_answer_relevancy.append(r["rag_ragas"]["answer_relevancy"])
        rag_context_precision.append(r["rag_ragas"]["context_precision"])
        rag_context_recall.append(r["rag_ragas"]["context_recall"])
        rag_em.append(r["rag_em"])
        rag_f1.append(r["rag_f1"])
        own_chunks.append(r["own_company_chunks"])
        baseline_answer_relevancy.append(r["baseline_ragas"]["answer_relevancy"])
        baseline_em.append(r["baseline_em"])
        baseline_f1.append(r["baseline_f1"])

    return {
        "n": len(rows),
        "rag": {
            "faithfulness": _mean(rag_faithfulness),
            "answer_relevancy": _mean(rag_answer_relevancy),
            "context_precision": _mean(rag_context_precision),
            "context_recall": _mean(rag_context_recall),
            "em": _mean(rag_em),
            "f1": _mean(rag_f1),
            "own_company_chunks": _mean(own_chunks),
        },
        "baseline": {
            "answer_relevancy": _mean(baseline_answer_relevancy),
            "em": _mean(baseline_em),
            "f1": _mean(baseline_f1),
        },
        # how many questions each RAGAS metric actually scored - a mean over 20
        # of 50 is not the same claim as a mean over 50
        "coverage": {
            "faithfulness": _coverage(rag_faithfulness),
            "answer_relevancy": _coverage(rag_answer_relevancy),
            "context_precision": _coverage(rag_context_precision),
            "context_recall": _coverage(rag_context_recall),
            "baseline_answer_relevancy": _coverage(baseline_answer_relevancy),
        },
    }


def _group_by(records: list[dict], field: str) -> dict:
    # distinct values, each kept once, then one aggregate per value
    values = []
    for r in records:
        if r[field] not in values:
            values.append(r[field])
    values.sort()

    grouped = {}
    for value in values:
        rows = []
        for r in records:
            if r[field] == value:
                rows.append(r)
        grouped[value] = agg(rows)
    return grouped


def summarise(records: list[dict]) -> dict:
    # overall, then split by question type and by company - a per-sector
    # breakdown is the point of picking five companies
    return {
        "overall": agg(records),
        "by_type": _group_by(records, "type"),
        "by_company": _group_by(records, "company"),
    }


def print_report(records: list[dict], summary: dict) -> None:
    print("\n" + "=" * 80)
    print("PER-QUESTION RESULTS")
    print("=" * 80)
    # "Own" = how many of the 5 chunks came from the right company
    header = (f"{'ID':<10}{'type':<10}{'Faith':>7}{'AnsRel':>8}{'CtxRel':>8}{'CtxRec':>8}"
              f"{'Own':>5}{'R-EM':>6}{'R-F1':>6}{'B-EM':>6}{'B-F1':>6}")
    print(header)
    print("-" * len(header))
    for r in records:
        g = r["rag_ragas"]   # keeps the print line below readable
        print(
            f"{r['id']:<10}{r['type']:<10}"
            f"{_fmt(g['faithfulness']):>7}{_fmt(g['answer_relevancy']):>8}"
            f"{_fmt(g['context_precision']):>8}{_fmt(g['context_recall']):>8}"
            f"{r['own_company_chunks']:>5.0f}"
            f"{r['rag_em']:>6.0f}{r['rag_f1']:>6.2f}"
            f"{r['baseline_em']:>6.0f}{r['baseline_f1']:>6.2f}"
        )

    print("\n" + "=" * 80)
    print("AGGREGATES (RAG vs Baseline)")
    print("=" * 80)
    _print_agg("OVERALL", summary["overall"])
    for qtype, agg_result in summary["by_type"].items():
        _print_agg(f"type={qtype}", agg_result)
    for company, agg_result in summary["by_company"].items():
        _print_agg(f"company={company}", agg_result)


def _print_agg(label: str, agg: dict) -> None:
    rag, base = agg["rag"], agg["baseline"]
    cov = agg["coverage"]
    print(f"\n{label}  (n={agg['n']})")
    print(f"  RAG      Faith={_fmt(rag['faithfulness'])}  AnsRel={_fmt(rag['answer_relevancy'])}"
          f"  CtxRel={_fmt(rag['context_precision'])}  CtxRec={_fmt(rag['context_recall'])}"
          f"  EM={_fmt(rag['em'])}  F1={_fmt(rag['f1'])}")
    # the count behind each RAGAS mean, not the number of questions asked
    print(f"           scored: Faith={cov['faithfulness']}  AnsRel={cov['answer_relevancy']}"
          f"  CtxRel={cov['context_precision']}  CtxRec={cov['context_recall']}")
    print(f"           OwnCompanyChunks={_fmt(rag['own_company_chunks'])}/5   (diagnostic, not a score)")
    print(f"  Baseline AnsRel={_fmt(base['answer_relevancy'])}  EM={_fmt(base['em'])}  F1={_fmt(base['f1'])}"
          f"   (context metrics N/A - baseline has no retrieval)")


def _fmt(x) -> str:
    # dash when there's no score
    if x is None:
        return "  -  "
    else:
        return f"{x:.2f}"


# main #
def main():
    parser = argparse.ArgumentParser(description="Score RAG against the baseline.")
    parser.add_argument("--run-name", default="all", help="prefix for the files in results/")
    parser.add_argument("--qa-file", default=QA_FILE)
    parser.add_argument("--company", default=None, help="only score one company's questions")
    args = parser.parse_args()

    with open(args.qa_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    if args.company:
        wanted = []
        for qa in qa_pairs:
            if qa["company"] == args.company:
                wanted.append(qa)
        qa_pairs = wanted

    print(f"Loaded {len(qa_pairs)} QA pairs from {args.qa_file}.\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = f"{RESULTS_DIR}/{args.run_name}_records.jsonl"

    # ask -> grade -> average -> print
    records = run_systems(qa_pairs, checkpoint_path)
    run_ragas(records)
    # scores go back into the checkpoint so the next run only patches the gaps
    save_checkpoint(records, checkpoint_path)
    summary = summarise(records)
    print_report(records, summary)

    with open(f"{RESULTS_DIR}/{args.run_name}_results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    with open(f"{RESULTS_DIR}/{args.run_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved detailed results to {RESULTS_DIR}/{args.run_name}_results.json")
    print(f"Saved aggregates to {RESULTS_DIR}/{args.run_name}_summary.json")


if __name__ == "__main__":
    main()
